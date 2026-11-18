"""Training-only feature fitting, metrics, and validation-only selection."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def metrics(y, predicted) -> dict:
    y, predicted = np.asarray(y, dtype=int), np.asarray(predicted, dtype=int)
    if y.shape != predicted.shape or not np.isin(y, [0, 1]).all() or not np.isin(predicted, [0, 1]).all():
        raise ValueError("Labels and predictions must be aligned binary arrays.")
    tp = int(((y == 1) & (predicted == 1)).sum())
    fp = int(((y == 0) & (predicted == 1)).sum())
    tn = int(((y == 0) & (predicted == 0)).sum())
    fn = int(((y == 1) & (predicted == 0)).sum())
    return {"n_messages": len(y), "n_spam": tp + fn, "n_ham": tn + fp,
            "spam_precision": tp / (tp + fp) if tp + fp else 0.0,
            "spam_recall": tp / (tp + fn) if tp + fn else 0.0,
            "spam_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "ham_fpr": fp / (fp + tn) if fp + tn else np.nan,
            "accuracy": (tp + tn) / len(y) if len(y) else np.nan,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def build_vectorizer(representation: str, config: dict) -> TfidfVectorizer:
    shared = dict(lowercase=True, strip_accents=None, stop_words=None,
                  min_df=config["min_df"], max_df=1.0, max_features=config["max_features"],
                  norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False, dtype=np.float64)
    if representation == "word":
        return TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w\w+\b",
                               ngram_range=tuple(config["word_ngram_range"]), **shared)
    if representation == "character":
        return TfidfVectorizer(analyzer="char", ngram_range=tuple(config["character_ngram_range"]), **shared)
    raise ValueError(representation)


def training_features(train: pd.DataFrame, representation: str, config: dict,
                      augmentation: pd.DataFrame | None = None):
    texts, y = train.text.tolist(), train.y.to_numpy()
    if augmentation is not None:
        if train.source_id.tolist() != augmentation.source_id.tolist():
            raise ValueError("Augmented copies are not aligned with training sources.")
        if not np.array_equal(train.y, augmentation.y):
            raise ValueError("Augmented labels differ from their sources.")
        texts += augmentation.text.tolist()
        y = np.concatenate([y, y])
        weights = np.full(len(texts), 0.5)
    else:
        weights = np.ones(len(texts))
    vectorizer = build_vectorizer(representation, config)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix, y, weights


def fit_classifier(matrix, y, weights, c_value: float, seed: int, config: dict):
    # Omitting penalty uses L2 in both sklearn 1.4-1.7 and 1.8-1.9,
    # without invoking the deprecated penalty argument in recent sklearn.
    classifier = LogisticRegression(C=c_value, solver="liblinear", max_iter=config["max_iter"],
                                    tol=config["tolerance"], class_weight=None, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(matrix, y, sample_weight=weights)
    if classifier.classes_.tolist() != [0, 1]:
        raise ValueError("Training must contain both ham and spam.")
    return classifier


def predict(bundle: dict, frame: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    threshold = bundle["threshold"] if threshold is None else threshold
    output = frame.copy()
    output["spam_probability"] = bundle["classifier"].predict_proba(
        bundle["vectorizer"].transform(frame.text))[:, 1]
    output["predicted_y"] = (output.spam_probability >= threshold).astype(int)
    output["predicted_label"] = np.where(output.predicted_y == 1, "spam", "ham")
    output["threshold"] = threshold
    output["correct"] = output.predicted_y == output.y
    return output


def select_threshold(y, probabilities, thresholds) -> tuple[float, dict]:
    # Fixed tie-break: higher F1, lower ham FPR, higher recall, closer to 0.5,
    # then higher threshold. Selection never sees unseen swaps or test data.
    candidates = []
    for threshold in thresholds:
        score = metrics(y, np.asarray(probabilities) >= threshold)
        key = (score["spam_f1"], -score["ham_fpr"], score["spam_recall"],
               -abs(threshold - 0.5), threshold)
        candidates.append((key, threshold, score))
    _, threshold, score = max(candidates, key=lambda row: row[0])
    return float(threshold), score


def low_fpr_threshold(y, probabilities, target: float) -> float:
    """Lowest inclusive >= cutoff retaining at most floor(target*n_ham) false positives."""
    if not 0 <= target < 1:
        raise ValueError("FPR target must lie in [0, 1).")
    ham = np.sort(np.asarray(probabilities)[np.asarray(y) == 0])
    if len(ham) == 0:
        raise ValueError("Clean validation contains no ham.")
    allowed = int(np.floor(target * len(ham)))
    return float(np.nextafter(ham[-allowed - 1], np.inf))
