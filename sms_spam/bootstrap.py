"""Paired test-message bootstrap; all variants and fitted seeds stay paired."""
from __future__ import annotations

import numpy as np
import pandas as pd


def paired_bootstrap(predictions: pd.DataFrame, replicates: int, seed: int,
                     unit: str = "message") -> pd.DataFrame:
    frame = predictions[predictions.operating_point == "selected_f1"]
    seeds = sorted(frame.seed.unique())
    models = sorted(frame.model.unique())
    conditions = list(dict.fromkeys(frame.condition))
    first = frame[(frame.seed == seeds[0]) & (frame.model == models[0]) & (frame.condition == "clean")]
    sources = sorted(first.source_id)
    first = first.set_index("source_id").loc[sources]
    y = first.y.to_numpy(dtype=int)
    predicted = np.empty((len(models), len(seeds), len(conditions), len(sources)), dtype=float)
    for m, model in enumerate(models):
        for s, split_seed in enumerate(seeds):
            for c, condition in enumerate(conditions):
                subset = frame[(frame.model == model) & (frame.seed == split_seed) & (frame.condition == condition)]
                if subset.source_id.duplicated().any() or set(subset.source_id) != set(sources):
                    raise ValueError("Bootstrap requires one prediction per paired source/model/seed/condition.")
                subset = subset.set_index("source_id").loc[sources]
                if not np.array_equal(y, subset.y):
                    raise ValueError("Labels differ across paired predictions.")
                predicted[m, s, c] = subset.predicted_y.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    if unit == "message":
        weights = rng.multinomial(len(sources), np.full(len(sources), 1 / len(sources)), size=replicates).T
    elif unit == "group":
        codes, groups = pd.factorize(first.group_id, sort=True)
        group_weights = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=replicates).T
        weights = group_weights[codes]
    else:
        raise ValueError("Bootstrap unit must be message or group.")
    weights = np.column_stack([np.ones(len(sources)), weights]).astype(float)
    flat = predicted.reshape(-1, len(sources))
    tp, fp = (flat * y) @ weights, (flat * (1 - y)) @ weights
    positive, negative = y @ weights, (1 - y) @ weights

    def divide(a, b, missing=0.0):
        return np.divide(a, b, out=np.full_like(a, missing), where=b != 0)

    raw = {"spam_f1": divide(2 * tp, tp + fp + positive),
           "spam_precision": divide(tp, tp + fp), "spam_recall": divide(tp, positive),
           "ham_fpr": divide(fp, negative, np.nan)}
    values = {key: array.reshape(len(models), len(seeds), len(conditions), -1).mean(axis=1)
              for key, array in raw.items()}
    clean_index = conditions.index("clean")
    values["f1_drop"] = values["spam_f1"][:, clean_index:clean_index + 1, :] - values["spam_f1"]
    contrasts = [("character_baseline", "word_baseline"),
                 ("word_augmented", "word_baseline"),
                 ("character_augmented", "character_baseline")]
    rows = []
    for candidate, reference in contrasts:
        if candidate not in models or reference not in models:
            continue
        for metric, array in values.items():
            difference = array[models.index(candidate)] - array[models.index(reference)]
            for i, condition in enumerate(conditions):
                draws = difference[i, 1:]
                valid = draws[np.isfinite(draws)]
                low, high = np.quantile(valid, [0.025, 0.975]) if len(valid) else (np.nan, np.nan)
                rows.append({"candidate": candidate, "reference": reference, "condition": condition,
                             "metric": metric, "difference_pp": difference[i, 0] * 100,
                             "ci95_low_pp": low * 100, "ci95_high_pp": high * 100,
                             "replicates": replicates, "valid_replicates": len(valid),
                             "bootstrap_unit": unit, "n_sources": len(sources), "n_fitted_seeds": len(seeds)})
    return pd.DataFrame(rows)
