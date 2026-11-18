"""Download, audit, group duplicate messages, and fix leakage-resistant splits."""
from __future__ import annotations

import io
import re
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import normalize

from .common import environment, file_hash, object_hash, read_csv, read_json, write_csv, write_json

DATASET_PAGE = "https://archive.ics.uci.edu/dataset/228/sms+spam+collection"
DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/228/sms%2Bspam%2Bcollection.zip"
PREPARATION_KEYS = ("test_fraction", "validation_fraction", "test_seed", "split_seeds",
                    "split_candidates", "near_duplicate_similarity")


def download_data(root: Path) -> Path:
    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    target = raw / "SMSSpamCollection"
    if target.is_file():
        print(f"Using existing data: {target}", flush=True)
        return target
    archive = raw / "sms_spam_collection.zip"
    if archive.exists():
        payload = archive.read_bytes()
    else:
        request = urllib.request.Request(DOWNLOAD_URL, headers={"User-Agent": "ECE783-SMS-research/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read(10_000_001)
            if len(payload) > 10_000_000:
                raise ValueError("Unexpected download size; expected the small UCI SMS archive.")
        except Exception as exc:
            raise RuntimeError(f"Download failed. Open {DATASET_PAGE}, click Download, then place "
                               f"SMSSpamCollection in {raw}. Original error: {exc}") from exc
    # Read only two named members. Never extract arbitrary paths from a ZIP.
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        members = {Path(n).name.lower(): n for n in zf.namelist()}
        if "smsspamcollection" not in members:
            raise ValueError("Archive does not contain SMSSpamCollection.")
        for key, filename in [("smsspamcollection", "SMSSpamCollection"), ("readme", "readme")]:
            if key in members:
                info = zf.getinfo(members[key])
                if info.file_size > 10_000_000:
                    raise ValueError("Unexpected archive member size.")
                (raw / filename).write_bytes(zf.read(members[key]))
    archive.write_bytes(payload)
    write_json(raw / "download_manifest.json", {
        "dataset_page": DATASET_PAGE, "download_url": DOWNLOAD_URL,
        "archive_sha256": file_hash(archive), "data_sha256": file_hash(target),
        "citation": "Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection. UCI. DOI: 10.24432/C5CC84",
    })
    return target


def load_raw(path: Path) -> pd.DataFrame:
    rows = []
    # Split specifically on newline, not all Unicode separators inside messages.
    for line_no, line in enumerate(path.read_bytes().decode("utf-8-sig").split("\n"), 1):
        line = line.removesuffix("\r")
        if not line:
            continue
        label, sep, message = line.partition("\t")
        if not sep or label not in {"ham", "spam"} or not message.strip():
            raise ValueError(f"Invalid UCI record on physical line {line_no}.")
        rows.append({"source_id": f"sms_{line_no:05d}", "raw_line": line_no,
                     "label": label, "y": int(label == "spam"), "text": message})
    if not rows:
        raise ValueError("The input file contains no SMS records.")
    return pd.DataFrame(rows)


def remove_exact_duplicates(frame: pd.DataFrame):
    conflicts = frame.groupby("text", sort=False)["y"].nunique()
    ambiguous = set(conflicts[conflicts > 1].index)
    conflict_rows = frame[frame.text.isin(ambiguous)].copy()
    usable = frame[~frame.text.isin(ambiguous)]
    duplicated = usable.duplicated("text", keep="first")
    duplicate_rows = usable[duplicated].copy()
    # Exact means identical message text, including case, punctuation and spaces.
    unique = usable[~duplicated].reset_index(drop=True).copy()
    return unique, duplicate_rows, conflict_rows


def grouping_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def near_duplicate_groups(frame: pd.DataFrame, threshold: float, batch_size: int = 256):
    """Connected components of cosine >= threshold on binary char_wb 3-5 grams.

    This separate, non-IDF representation is used only for partitioning. It is
    discarded before predictive TF-IDF is fitted on training messages.
    """
    n = len(frame)
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    texts = [grouping_text(t) for t in frame.text]
    seen = {}
    edges = []
    for i, text in enumerate(texts):
        if text in seen:
            j = seen[text]
            union(i, j)
            edges.append((frame.source_id.iat[j], frame.source_id.iat[i], 1.0, "normalized_exact"))
        else:
            seen[text] = i
    vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=(3, 5), binary=True,
                                 lowercase=False, dtype=np.float64)
    matrix = normalize(vectorizer.fit_transform(texts), norm="l2")
    for start in range(0, n, batch_size):
        similarities = (matrix[start:start + batch_size] @ matrix.T).tocoo()
        keep = ((similarities.data >= threshold - 1e-12)
                & (similarities.col > similarities.row + start))
        for row, col, similarity in zip(similarities.row[keep] + start,
                                        similarities.col[keep], similarities.data[keep]):
            union(int(row), int(col))
            edges.append((frame.source_id.iat[row], frame.source_id.iat[col],
                          float(min(1.0, similarity)), "cosine"))
    groups = [f"group_{find(i):05d}" for i in range(n)]
    return groups, pd.DataFrame(edges, columns=["source_a", "source_b", "similarity", "rule"])


def group_holdout(frame: pd.DataFrame, fraction: float, seed: int, candidates: int):
    """Choose the split nearest target message and per-class counts, without model scores."""
    splitter = GroupShuffleSplit(n_splits=candidates, test_size=fraction, random_state=seed)
    class_counts = np.bincount(frame.y.to_numpy(), minlength=2)
    if (class_counts < 2).any():
        raise ValueError("Need at least two messages of each class for a holdout.")
    best = None
    for train_idx, held_idx in splitter.split(frame, frame.y, frame.group_id):
        held_counts = np.bincount(frame.y.iloc[held_idx], minlength=2)
        if (held_counts == 0).any() or (held_counts == class_counts).any():
            continue
        score = float((len(held_idx) / len(frame) - fraction) ** 2
                      + np.square(held_counts / class_counts - fraction).sum())
        if best is None or score < best[0]:
            best = (score, train_idx, held_idx)
    if best is None:
        raise ValueError("Cannot form a group-disjoint split containing both classes.")
    return frame.iloc[best[1]].copy(), frame.iloc[best[2]].copy()


def assert_disjoint(*frames: pd.DataFrame) -> None:
    for i, left in enumerate(frames):
        for right in frames[i + 1:]:
            for key in ["source_id", "group_id"]:
                if set(left[key]) & set(right[key]):
                    raise AssertionError(f"Cross-partition leakage detected for {key}.")


def partition_counts(named_frames) -> pd.DataFrame:
    return pd.DataFrame([{"partition": name, "total_messages": len(df),
                          "spam": int(df.y.sum()), "ham": int((df.y == 0).sum()),
                          "groups": int(df.group_id.nunique())}
                         for name, df in named_frames])


def length_statistics(named_frames) -> pd.DataFrame:
    rows = []
    for name, df in named_frames:
        for unit, lengths in [("characters", df.text.str.len()),
                              ("whitespace_tokens", df.text.map(lambda t: len(t.split())))]:
            rows.append({"partition": name, "unit": unit, "n": len(df),
                         "median": float(np.median(lengths)),
                         "p95": float(np.quantile(lengths, 0.95)), "max": int(max(lengths))})
    return pd.DataFrame(rows)


def prepare(root: Path, config: dict, allow_download: bool = False) -> dict:
    raw = root / "data" / "raw" / "SMSSpamCollection"
    if not raw.exists():
        if allow_download:
            download_data(root)
        else:
            raise FileNotFoundError(f"Put SMSSpamCollection in {raw.parent}, or add --download.")
    out = root / "data" / "processed"
    settings = {k: config[k] for k in PREPARATION_KEYS}
    fingerprint = object_hash({"raw_sha256": file_hash(raw), "settings": settings,
                               "data_module_sha256": file_hash(Path(__file__))})
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["preparation_fingerprint"] != fingerprint:
            raise ValueError("Data/split settings changed. Use a new --workspace directory to keep "
                             "the existing test partition fixed; do not silently resplit.")
        for relative, expected in manifest["file_sha256"].items():
            path = out / relative
            if not path.exists() or file_hash(path) != expected:
                raise ValueError(f"Prepared data changed or missing: {path}")
        print("Reusing verified fixed data partitions.", flush=True)
        return manifest
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Incomplete preparation exists at {out}. Use a new --workspace or "
                         "inspect and move that incomplete directory before retrying.")
    print("Auditing exact duplicates and grouping near-duplicates...", flush=True)
    original = load_raw(raw)
    unique, duplicates, conflicts = remove_exact_duplicates(original)
    unique["group_id"], edges = near_duplicate_groups(unique, config["near_duplicate_similarity"])
    development, test = group_holdout(unique, config["test_fraction"], config["test_seed"],
                                     config["split_candidates"])
    write_csv(test, out / "test_reserved.csv")
    write_csv(duplicates, out / "exact_duplicates_removed.csv")
    write_csv(conflicts, out / "conflicting_labels_removed.csv")
    write_csv(edges, out / "near_duplicate_edges.csv")
    counts, memberships, length_frames = [], [], [("raw", original), ("deduplicated", unique)]
    for seed in config["split_seeds"]:
        train, validation = group_holdout(
            development, config["validation_fraction"] / (1 - config["test_fraction"]),
            seed, config["split_candidates"])
        assert_disjoint(train, validation, test)
        named = [("train", train), ("validation", validation), ("test", test)]
        counts.append(partition_counts(named).assign(seed=seed))
        for name, df in named:
            memberships.append(df[["source_id", "group_id", "label"]].assign(seed=seed, partition=name))
        write_csv(train, out / f"train_seed_{seed}.csv")
        write_csv(validation, out / f"validation_seed_{seed}.csv")
        if seed == config["split_seeds"][0]:
            length_frames.extend(named)
    write_csv(pd.concat(counts, ignore_index=True), out / "split_counts.csv")
    write_csv(pd.concat(memberships, ignore_index=True), out / "split_membership.csv")
    write_csv(length_statistics(length_frames), out / "length_statistics.csv")
    audit = {"raw_messages": len(original), "raw_spam": int(original.y.sum()),
             "raw_ham": int((original.y == 0).sum()), "unique_messages": len(unique),
             "exact_duplicate_rows_removed": len(duplicates), "conflict_rows_removed": len(conflicts),
             "groups": int(unique.group_id.nunique()),
             "largest_group": int(unique.groupby("group_id").size().max()),
             "mixed_label_groups": int((unique.groupby("group_id").y.nunique() > 1).sum()),
             "near_duplicate_edge_records": len(edges),
             "grouping_method": "NFKC + casefold + collapsed whitespace; binary char_wb 3-5 "
                                "grams; cosine >= threshold; connected components; no IDF",
             "similarity_threshold": config["near_duplicate_similarity"],
             "cross_partition_group_overlap": 0,
             "predictive_test_evaluation_performed": False}
    write_json(out / "data_audit.json", audit)
    manifest = environment(root, config, "prepare")
    manifest.update({"preparation_fingerprint": fingerprint, "raw_sha256": file_hash(raw),
                     "dataset_page": DATASET_PAGE, "settings": settings, "audit": audit,
                     "file_sha256": {p.name: file_hash(p) for p in sorted(out.glob("*")) if p.is_file()}})
    write_json(manifest_path, manifest)
    print(f"Prepared {len(unique)} distinct messages in {audit['groups']} groups.", flush=True)
    return manifest


def load_development(root: Path, seed: int):
    out = root / "data" / "processed"
    return read_csv(out / f"train_seed_{seed}.csv"), read_csv(out / f"validation_seed_{seed}.csv")
