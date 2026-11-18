"""Label-blind, reproducible edits, made only after source messages are split."""
from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

SUBSTITUTIONS = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}
WORD_RE = re.compile(r"\b[A-Za-z]+\b")
EDIT_TYPES = ("substitution", "space", "swap")


def local_rng(seed: int, source_id: str, stream: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}|{source_id}|{stream}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "little"))


def edit_positions(word: str, kind: str) -> list[int]:
    if kind == "substitution":
        return [i for i, c in enumerate(word) if c.lower() in SUBSTITUTIONS]
    if kind == "space":
        return list(range(1, len(word))) if len(word) >= 4 else []
    if kind == "swap":
        return [i for i in range(len(word) - 1) if word[i].lower() != word[i + 1].lower()] if len(word) >= 4 else []
    raise ValueError(f"Unknown edit: {kind}")


def perturb_text(text: str, kind: str, probability: float, seed: int, source_id: str) -> dict:
    if kind not in EDIT_TYPES or not 0 <= probability <= 1:
        raise ValueError("Unknown edit type or probability outside [0, 1].")
    rng = local_rng(seed, source_id, kind)
    eligible = edited = 0

    def replace(match):
        nonlocal eligible, edited
        word = match.group(0)
        positions = edit_positions(word, kind)
        if not positions:
            return word
        eligible += 1
        # Draw position even when not selected: rate sweeps have nested edited words.
        selected = rng.random() < probability
        position = int(rng.choice(positions))
        if not selected:
            return word
        edited += 1
        if kind == "substitution":
            return word[:position] + SUBSTITUTIONS[word[position].lower()] + word[position + 1:]
        if kind == "space":
            return word[:position] + " " + word[position:]
        return (word[:position] + word[position + 1] + word[position] + word[position + 2:])

    result = WORD_RE.sub(replace, text)
    return {"text": result, "eligible_words": eligible, "edited_words": edited,
            "changed": result != text, "word_count": len(WORD_RE.findall(text))}


def make_condition(frame: pd.DataFrame, kind: str, probability: float, seed: int) -> pd.DataFrame:
    rows = []
    for row in frame.itertuples(index=False):
        if kind == "clean":
            result = {"text": row.text, "eligible_words": 0, "edited_words": 0,
                      "changed": False, "word_count": len(WORD_RE.findall(row.text))}
        else:
            result = perturb_text(row.text, kind, probability, seed, row.source_id)
        rows.append({"source_id": row.source_id, "group_id": row.group_id, "label": row.label,
                     "y": row.y, "original_text": row.text, "condition": kind,
                     "edit_probability": 0.0 if kind == "clean" else probability, **result})
    return pd.DataFrame(rows)


def make_augmentation(train: pd.DataFrame, probability: float, seed: int) -> pd.DataFrame:
    rows = []
    for row in train.itertuples(index=False):
        # One edit type per source message, selected with probability 1/2.
        kind = str(local_rng(seed, row.source_id, "augmentation_type").choice(["substitution", "space"]))
        result = perturb_text(row.text, kind, probability, seed, row.source_id)
        rows.append({"source_id": row.source_id, "group_id": row.group_id, "label": row.label,
                     "y": row.y, "original_text": row.text, "condition": kind,
                     "edit_probability": probability, **result})
    return pd.DataFrame(rows)


def edit_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in ["all", "ham", "spam"]:
        subset = frame if label == "all" else frame[frame.label == label]
        eligible, edited = int(subset.eligible_words.sum()), int(subset.edited_words.sum())
        changed = int(subset.changed.sum())
        rows.append({"label": label, "n_messages": len(subset), "n_changed": changed,
                     "fraction_changed": changed / len(subset) if len(subset) else np.nan,
                     "eligible_words": eligible, "edited_words": edited,
                     "realized_edit_rate": edited / eligible if eligible else np.nan,
                     "messages_without_eligible_words": int((subset.eligible_words == 0).sum())})
    return pd.DataFrame(rows)
