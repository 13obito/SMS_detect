"""Separate the midsemester baseline, validation tuning, and frozen test stage."""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .bootstrap import paired_bootstrap
from .common import environment, file_hash, read_csv, read_json, source_hashes, write_csv, write_json
from .data import load_development, prepare
from .model import fit_classifier, low_fpr_threshold, metrics, predict, select_threshold, training_features
from .perturb import edit_statistics, make_augmentation, make_condition
from .reporting import (aggregate_scores, changed_source_comparison, error_slices,
                        export_midsemester, final_figures, robustness_changes)

CONFIGURATIONS = [("word_baseline", "word", False), ("character_baseline", "character", False),
                  ("word_augmented", "word", True), ("character_augmented", "character", True)]


def stage_path(root: Path, name: str) -> Path:
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("Run names may contain only letters, digits, underscores and hyphens.")
    return root / "results" / name


def run_midsemester(root: Path, config: dict, download: bool = False, compare: bool = False,
                    run_name: str | None = None) -> Path:
    prepared = prepare(root, config, download)
    seed = config["split_seeds"][0]
    train, validation = load_development(root, seed)
    out = stage_path(root, run_name or ("representation_comparison" if compare else "midsemester"))
    out.mkdir(parents=True, exist_ok=True)
    conditions = {kind: make_condition(validation, kind, config["edit_probability"], config["evaluation_edit_seed"])
                  for kind in ["clean", "substitution"]}
    scores, predictions, info = [], [], []
    for name, representation, _ in CONFIGURATIONS[:2 if compare else 1]:
        print(f"Fitting {name} on {len(train)} original training messages...", flush=True)
        started = time.perf_counter()
        vectorizer, matrix, y, weights = training_features(train, representation, config)
        classifier = fit_classifier(matrix, y, weights, config["baseline_C"], seed, config)
        bundle = {"vectorizer": vectorizer, "classifier": classifier, "threshold": config["baseline_threshold"]}
        joblib.dump(bundle, out / f"{name}.joblib")
        info.append({"model": name, "training_sources": len(train), "training_rows": len(y),
                     "training_weight_sum": float(weights.sum()), "vocabulary_size": len(vectorizer.vocabulary_),
                     "classifier_iterations": int(classifier.n_iter_.max()),
                     "fit_seconds": time.perf_counter() - started, "seed": seed})
        for kind, frame in conditions.items():
            result = predict(bundle, frame).assign(model=name, seed=seed, partition="validation")
            predictions.append(result)
            scores.append({"model": name, "condition": kind, "seed": seed, "partition": "validation",
                           **metrics(result.y, result.predicted_y)})
    prediction_frame, score_frame = pd.concat(predictions, ignore_index=True), pd.DataFrame(scores)
    export_midsemester(root, out, config, seed, score_frame, prediction_frame, conditions, info)
    write_csv(robustness_changes(score_frame), out / "robustness_changes.csv")
    manifest = environment(root, config, "representation_comparison" if compare else "midsemester")
    manifest.update({"data_fingerprint": prepared["preparation_fingerprint"], "test_predictions_computed": False,
                     "models": info})
    write_json(out / "run_manifest.json", manifest)
    print(f"Completed validation-only run: {out}", flush=True)
    return out


def tune(root: Path, config: dict, download: bool = False, run_name: str = "frozen_development",
         low_fpr: bool = False, rates: list[float] | None = None, bootstrap_unit: str = "message") -> Path:
    prepared = prepare(root, config, download)
    rates = list(dict.fromkeys([config["edit_probability"]] if rates is None else rates))
    if not rates or any(not 0 < p <= 1 for p in rates):
        raise ValueError("Predeclared final edit rates must be in (0, 1].")
    out = stage_path(root, run_name)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Run already exists: {out}. Choose a new --run-name to preserve frozen settings.")
    out.mkdir(parents=True, exist_ok=True)
    models_dir = out / "models"
    models_dir.mkdir()
    selected, trials, validation_scores, validation_predictions, edit_stats = [], [], [], [], []
    for seed in config["split_seeds"]:
        train, validation = load_development(root, seed)
        conditions = {kind: make_condition(validation, kind, config["edit_probability"], config["evaluation_edit_seed"])
                      for kind in ["clean", "substitution", "space"]}
        # Each source appears exactly once in each condition: a 1/3-1/3-1/3 mixture.
        mixture = pd.concat(conditions.values(), ignore_index=True)
        augmentation = make_augmentation(train, config["edit_probability"], seed + config["augmentation_seed_offset"])
        write_csv(augmentation, out / f"training_augmentation_seed_{seed}.csv")
        for kind, frame in conditions.items():
            if kind != "clean":
                edit_stats.append(edit_statistics(frame).assign(seed=seed, condition=kind, partition="validation"))
        for kind in ["substitution", "space"]:
            edit_stats.append(edit_statistics(augmentation[augmentation.condition == kind]).assign(
                seed=seed, condition=kind, partition="training_augmentation"))
        for name, representation, augmented in CONFIGURATIONS:
            print(f"Validation selection: seed={seed}, model={name}", flush=True)
            vectorizer, matrix, y, weights = training_features(
                train, representation, config, augmentation if augmented else None)
            val_matrix = vectorizer.transform(mixture.text)
            best = None
            for c_value in config["C_candidates"]:
                classifier = fit_classifier(matrix, y, weights, c_value, seed, config)
                probabilities = classifier.predict_proba(val_matrix)[:, 1]
                threshold, score = select_threshold(mixture.y, probabilities, config["threshold_candidates"])
                for candidate in config["threshold_candidates"]:
                    trials.append({"seed": seed, "model": name, "C": c_value, "threshold": candidate,
                                   **metrics(mixture.y, probabilities >= candidate)})
                key = (score["spam_f1"], -score["ham_fpr"], score["spam_recall"],
                       -abs(threshold - 0.5), threshold, -c_value)
                if best is None or key > best[0]:
                    best = (key, classifier, c_value, threshold, score)
            _, classifier, c_value, threshold, score = best
            bundle = {"vectorizer": vectorizer, "classifier": classifier, "threshold": threshold}
            low_threshold = None
            if low_fpr:
                clean = predict(bundle, conditions["clean"])
                low_threshold = low_fpr_threshold(clean.y, clean.spam_probability, config["low_fpr_target"])
                bundle["low_fpr_threshold"] = low_threshold
            relative = f"models/{name}_seed_{seed}.joblib"
            joblib.dump(bundle, out / relative)
            selected.append({"seed": seed, "model": name, "representation": representation,
                             "augmentation": augmented, "C": c_value, "threshold": threshold,
                             "low_fpr_threshold": low_threshold, "validation_mixture": score,
                             "n_training_sources": len(train), "n_training_rows": len(y),
                             "training_weight_sum": float(weights.sum()), "n_validation_sources": len(validation),
                             "n_validation_variant_rows": len(mixture), "vocabulary_size": len(vectorizer.vocabulary_),
                             "model_file": relative, "model_sha256": file_hash(out / relative)})
            for kind, frame in conditions.items():
                result = predict(bundle, frame).assign(seed=seed, model=name, operating_point="selected_f1")
                validation_predictions.append(result)
                validation_scores.append({"seed": seed, "model": name, "condition": kind,
                                          "operating_point": "selected_f1", **metrics(result.y, result.predicted_y)})
                if low_fpr:
                    result_low = predict(bundle, frame, low_threshold).assign(seed=seed, model=name, operating_point="low_fpr")
                    validation_predictions.append(result_low)
                    validation_scores.append({"seed": seed, "model": name, "condition": kind,
                                              "operating_point": "low_fpr", **metrics(result_low.y, result_low.predicted_y)})
    write_csv(pd.DataFrame(trials), out / "validation_selection_trials.csv")
    score_frame = pd.DataFrame(validation_scores)
    write_csv(score_frame, out / "validation_metrics_by_seed.csv")
    write_csv(aggregate_scores(score_frame, ["model", "condition", "operating_point"]), out / "validation_mean_std.csv")
    write_csv(pd.concat(validation_predictions, ignore_index=True), out / "validation_predictions.csv")
    write_csv(pd.concat(edit_stats, ignore_index=True), out / "perturbation_statistics.csv")
    frozen = environment(root, config, "frozen_validation_selection")
    frozen.update({"data_fingerprint": prepared["preparation_fingerprint"],
                   "prepared_manifest_sha256": file_hash(root / "data" / "processed" / "manifest.json"),
                   "test_sha256": prepared["file_sha256"]["test_reserved.csv"],
                   "test_edit_rates": rates, "bootstrap_unit": bootstrap_unit, "low_fpr_enabled": low_fpr,
                   "test_predictions_computed": False, "unseen_swap_used_for_selection": False,
                   "selection_objective": "pooled spam F1 on equally represented clean/substitution/space variants",
                   "tie_break": "max F1, min ham FPR, max recall, threshold closest to .5, higher threshold, smaller C",
                   "models": selected})
    write_json(out / "frozen_manifest.json", frozen)
    print(f"Frozen all settings without evaluating the test set: {out}", flush=True)
    return out


def final_evaluation(root: Path, frozen_name: str) -> Path:
    run = stage_path(root, frozen_name)
    frozen = read_json(run / "frozen_manifest.json")
    config = frozen["config"]
    if frozen["source_sha256"] != source_hashes(root):
        raise ValueError("Executable source changed since settings were frozen. Restore that source or create a new frozen run.")
    current_packages = environment(root, config, "verification")["packages"]
    if current_packages != frozen["packages"]:
        raise ValueError("Package versions changed since model freezing. Use the same Python environment.")
    processed = root / "data" / "processed"
    if file_hash(processed / "manifest.json") != frozen["prepared_manifest_sha256"]:
        raise ValueError("Prepared dataset manifest changed after freezing.")
    if file_hash(processed / "test_reserved.csv") != frozen["test_sha256"]:
        raise ValueError("Reserved test file changed after freezing.")
    for item in frozen["models"]:
        if file_hash(run / item["model_file"]) != item["model_sha256"]:
            raise ValueError("A frozen model file was modified.")
    test = read_csv(processed / "test_reserved.csv")
    out = run / "final_test"
    out.mkdir(exist_ok=True)
    conditions = {"clean": make_condition(test, "clean", config["edit_probability"], config["evaluation_edit_seed"])}
    for rate in frozen["test_edit_rates"]:
        for kind in ["substitution", "space", "swap"]:
            condition = f"{kind}_p{rate:g}"
            conditions[condition] = make_condition(test, kind, rate, config["evaluation_edit_seed"]).assign(condition=condition)
    predictions, scores = [], []
    for item in frozen["models"]:
        # Load only locally produced, hash-verified files. Pickles from strangers are not supported.
        bundle = joblib.load(run / item["model_file"])
        print(f"Frozen test evaluation: seed={item['seed']}, model={item['model']}", flush=True)
        for kind, frame in conditions.items():
            for operating in ["selected_f1", *(["low_fpr"] if frozen["low_fpr_enabled"] else [])]:
                threshold = bundle["threshold"] if operating == "selected_f1" else bundle["low_fpr_threshold"]
                result = predict(bundle, frame, threshold).assign(seed=item["seed"], model=item["model"],
                                                                  operating_point=operating, partition="test")
                predictions.append(result)
                scores.append({"seed": item["seed"], "model": item["model"], "condition": kind,
                               "operating_point": operating, "threshold": threshold,
                               **metrics(result.y, result.predicted_y)})
    prediction_frame, score_frame = pd.concat(predictions, ignore_index=True), pd.DataFrame(scores)
    write_csv(prediction_frame, out / "test_predictions.csv")
    write_csv(score_frame, out / "test_metrics_by_seed.csv")
    summary = aggregate_scores(score_frame, ["model", "condition", "operating_point"])
    write_csv(summary, out / "test_mean_std.csv")
    changes = robustness_changes(score_frame)
    write_csv(changes, out / "test_robustness_changes.csv")
    change_columns = [c for c in changes if c.endswith("_change_pp") or c == "f1_drop_pp"]
    delta_summary = changes.groupby(["model", "condition", "operating_point"])[change_columns].agg(["mean", "std"])
    delta_summary.columns = [f"{a}_{b}" for a, b in delta_summary.columns]
    write_csv(delta_summary.reset_index(), out / "test_robustness_changes_mean_std.csv")
    write_csv(pd.concat([edit_statistics(frame).assign(condition=kind) for kind, frame in conditions.items()
                         if kind != "clean"], ignore_index=True), out / "test_perturbation_statistics.csv")
    write_csv(prediction_frame[~prediction_frame.correct], out / "test_errors.csv")
    write_csv(error_slices(prediction_frame), out / "test_error_slices.csv")
    write_csv(changed_source_comparison(prediction_frame), out / "test_changed_source_comparison.csv")
    print("Computing paired bootstrap intervals...", flush=True)
    intervals = paired_bootstrap(prediction_frame, config["bootstrap_replicates"], config["bootstrap_seed"],
                                 frozen["bootstrap_unit"])
    write_csv(intervals, out / "paired_bootstrap_contrasts.csv")
    final_figures(summary, out)
    manifest = environment(root, config, "final_test")
    manifest.update({"frozen_manifest_sha256": file_hash(run / "frozen_manifest.json"),
                     "n_test_sources": len(test), "test_edit_rates": frozen["test_edit_rates"],
                     "bootstrap_unit": frozen["bootstrap_unit"],
                     "bootstrap_interpretation": "95% percentile intervals for differences of mean seed-specific "
                     "metrics, paired over sources, all conditions, and all fitted runs; conditional on frozen fits. "
                     "Seed variation is reported separately as sample SD, not treated as independent test messages.",
                     "low_fpr_interpretation": "The empirical validation ham FPR target is not a test guarantee."})
    write_json(out / "run_manifest.json", manifest)
    print(f"Final test artifacts: {out}", flush=True)
    return out
