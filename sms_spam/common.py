"""File formats and reproducibility metadata shared by all stages."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """BOM allows Excel on Windows to recognize UTF-8; do not alter numeric precision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def read_csv(path: Path) -> pd.DataFrame:
    # Preserve legitimate message strings such as "NA" and leading/trailing spaces.
    return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)


def source_hashes(root: Path) -> dict:
    # --workspace redirects data/results, while executable source stays here.
    root = Path(__file__).resolve().parents[1]
    paths = [root / "run_experiments.py", *sorted((root / "sms_spam").glob("*.py"))]
    return {str(p.relative_to(root)).replace("\\", "/"): file_hash(p) for p in paths}


def environment(root: Path, config: dict, stage: str) -> dict:
    return {
        "stage": stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ["numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "joblib"]},
        "source_sha256": source_hashes(root),
        "config": config,
        "config_sha256": object_hash(config),
    }


def validate_config(c: dict) -> None:
    if not (0 < c["test_fraction"] < 1 and 0 < c["validation_fraction"] < 1
            and c["test_fraction"] + c["validation_fraction"] < 1):
        raise ValueError("Train, validation and test fractions must all be positive.")
    if not 0 < c["near_duplicate_similarity"] <= 1:
        raise ValueError("Near-duplicate cosine threshold must be in (0, 1].")
    if not 0 <= c["edit_probability"] <= 1:
        raise ValueError("Edit probability must be in [0, 1].")
    if not c["split_seeds"] or len(set(c["split_seeds"])) != len(c["split_seeds"]):
        raise ValueError("Provide at least one distinct split seed.")
    if any(v <= 0 for v in c["C_candidates"]) or c["baseline_C"] <= 0:
        raise ValueError("C values must be positive.")
    if not c["threshold_candidates"] or any(not 0 <= v <= 1 for v in c["threshold_candidates"]):
        raise ValueError("Selection thresholds must be in [0, 1].")
    if c["split_candidates"] < 1 or c["bootstrap_replicates"] < 1:
        raise ValueError("Split candidates and bootstrap replicates must be positive.")
