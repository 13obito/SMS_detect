"""Report-ready CSV tables, scientific figures, and auditable examples."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import object_hash, read_csv, read_json, write_csv, write_json
from .model import metrics
from .perturb import SUBSTITUTIONS, edit_statistics

METRICS = ["spam_precision", "spam_recall", "spam_f1", "ham_fpr"]
METRIC_LABELS = ["Spam precision", "Spam recall", "Spam F1", "Ham FPR"]


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def baseline_figures(scores: pd.DataFrame, out: Path) -> None:
    word = scores[scores.model == "word_baseline"].set_index("condition")
    if not {"clean", "substitution"}.issubset(word.index):
        return
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [3, 1]}, layout="constrained")
    colors = ["#2864a5", "#d17826"]
    x = np.arange(3)
    for i, (condition, label) in enumerate([("clean", "Clean"), ("substitution", "Substitution")]):
        values = 100 * word.loc[condition, METRICS[:3]].astype(float).to_numpy()
        bars = axes[0].bar(x + (i - 0.5) * 0.36, values, 0.36, label=label, color=colors[i])
        axes[0].bar_label(bars, fmt="%.1f", fontsize=9, padding=3)
        bar = axes[1].bar(i, float(word.loc[condition, "ham_fpr"]) * 100, color=colors[i])
        axes[1].bar_label(bar, fmt="%.2f", padding=3, fontsize=9)
    axes[0].set(xticks=x, xticklabels=METRIC_LABELS[:3], ylim=(0, 110), ylabel="Percent")
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False)
    maximum_fpr = max(0.5, float(word.ham_fpr.max()) * 100 * 1.35)
    axes[1].set(xticks=[0, 1], xticklabels=["Clean", "Edited"], ylim=(0, maximum_fpr),
                title="Ham false-positive rate", ylabel="Percent")
    n = int(word.loc["clean", "n_messages"])
    fig.suptitle(f"Word baseline on paired validation messages (n = {n})", fontsize=13)
    save_figure(fig, out / "figures" / "baseline_validation_metrics")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.7), layout="constrained")
    limit = max(float(word.tn.max()), float(word.tp.max()))
    for ax, condition in zip(axes, ["clean", "substitution"]):
        row = word.loc[condition]
        matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=limit)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center", fontsize=13,
                    color="white" if value > limit * 0.55 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Ham", "Spam"],
               yticklabels=["Ham", "Spam"], xlabel="Predicted label", ylabel="True label",
               title=condition.capitalize())
    fig.suptitle("Word baseline validation confusion matrices", fontsize=13)
    save_figure(fig, out / "figures" / "baseline_confusion_matrices")


def write_manual_review(edited: pd.DataFrame, out: Path, per_class: int, seed: int) -> Path:
    samples = []
    for label in ["ham", "spam"]:
        subset = edited[(edited.label == label) & edited.changed]
        samples.append(subset.sample(n=min(per_class, len(subset)), random_state=seed))
    sample = pd.concat(samples).sort_values(["label", "source_id"])
    columns = ["source_id", "label", "original_text", "text", "eligible_words", "edited_words"]
    sample = sample[columns].copy()
    identity = object_hash(sample.to_dict(orient="records"))[:12]
    path = out / "manual_review" / f"validation_substitution_{identity}.csv"
    if not path.exists():
        for name in ["readable_yes_no", "label_preserved_yes_no", "reviewer", "notes"]:
            sample[name] = ""
        write_csv(sample, path)
    # A repeat run never overwrites the human's notes or marks the task completed.
    return path


def error_slices(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = [key for key in ["seed", "model", "condition", "operating_point"] if key in predictions]
    for values, frame in predictions.groupby(keys, sort=False):
        identity = dict(zip(keys, values))
        condition = identity["condition"]
        slices = {
            "all": np.ones(len(frame), dtype=bool),
            "changed_only": frame.changed,
            "short_original_le_40_chars": frame.original_text.str.len() <= 40,
            "promotional_ham": (frame.y == 0) & frame.original_text.str.contains(
                r"\b(?:free|win|winner|prize|cash|offer|claim|urgent)\b", case=False, regex=True),
        }
        if condition != "clean":
            slices["few_eligible_words_le_2"] = frame.eligible_words <= 2
        for name, mask in slices.items():
            subset = frame[mask]
            if len(subset):
                rows.append({**identity, "slice": name,
                             **metrics(subset.y, subset.predicted_y)})
    return pd.DataFrame(rows)


def changed_source_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compare clean and edited predictions on exactly the same changed sources."""
    rows = []
    keys = [key for key in ["seed", "model", "operating_point"] if key in predictions]
    for values, frame in predictions.groupby(keys, sort=False):
        identity = dict(zip(keys, values))
        clean = frame[frame.condition == "clean"]
        for condition, edited in frame[frame.condition != "clean"].groupby("condition", sort=False):
            changed = edited[edited.changed]
            clean_subset = clean[clean.source_id.isin(changed.source_id)]
            if not len(changed):
                continue
            if len(changed) != len(clean_subset):
                raise ValueError("Changed-only comparisons must pair the same clean sources.")
            for evaluated, subset in [("clean", clean_subset), ("edited", changed)]:
                rows.append({**identity, "edit_condition": condition, "evaluated_text": evaluated,
                             **metrics(subset.y, subset.predicted_y)})
    return pd.DataFrame(rows)


def paired_cases(predictions: pd.DataFrame, out: Path) -> pd.DataFrame:
    word = predictions[predictions.model == "word_baseline"]
    clean = word[word.condition == "clean"]
    edited = word[word.condition == "substitution"]
    pairs = clean[["source_id", "label", "y", "text", "spam_probability", "predicted_label", "predicted_y"]].merge(
        edited[["source_id", "text", "spam_probability", "predicted_label", "predicted_y", "changed",
                "eligible_words", "edited_words"]], on="source_id", suffixes=("_clean", "_edited"), validate="1:1")
    pairs["probability_change"] = pairs.spam_probability_edited - pairs.spam_probability_clean
    pairs["prediction_flipped"] = pairs.predicted_y_clean != pairs.predicted_y_edited
    pairs["clean_correct_edited_wrong"] = (pairs.predicted_y_clean == pairs.y) & (pairs.predicted_y_edited != pairs.y)
    pairs["absolute_probability_change"] = pairs.probability_change.abs()
    pairs = pairs.sort_values(["clean_correct_edited_wrong", "prediction_flipped", "absolute_probability_change", "source_id"],
                              ascending=[False, False, False, True])
    write_csv(pairs, out / "paired_validation_predictions.csv")
    write_csv(pairs.head(20), out / "illustrative_cases.csv")
    write_json(out / "illustrative_cases.json", pairs.head(5).to_dict(orient="records"))
    return pairs


def export_midsemester(root: Path, out: Path, config: dict, seed: int,
                      scores: pd.DataFrame, predictions: pd.DataFrame,
                      conditions: dict, model_info: list) -> None:
    processed = root / "data" / "processed"
    counts = read_csv(processed / "split_counts.csv")
    table1 = counts[counts.seed == seed].drop(columns="seed")
    write_csv(table1, out / "table1_data_partitions.csv")
    write_csv(read_csv(processed / "length_statistics.csv"), out / "data_length_statistics.csv")
    settings = [
        ("Preprocessing", "lowercase; Unicode alphanumeric tokens of >=2 characters; punctuation separates word tokens; "
         "no stop-word removal, stemming, accent stripping, or truncation"),
        ("Word TF-IDF", f"ngram_range={config['word_ngram_range']}; min_df={config['min_df']}; "
         f"max_features={config['max_features']}; norm=l2; use_idf=True; smooth_idf=True; sublinear_tf=False"),
        ("Logistic regression", f"C={config['baseline_C']}; L2; solver=liblinear; max_iter={config['max_iter']}; "
         f"tol={config['tolerance']}; class_weight=None; fit_intercept=True"),
        ("Decision rule", f"predict spam when probability >= {config['baseline_threshold']}"),
        ("Seeds", f"test split={config['test_seed']}; train/validation split={seed}; "
         f"perturbation={config['evaluation_edit_seed']}; classifier={seed}"),
        ("Selection", "settings fixed before baseline evaluation; no baseline hyperparameter or threshold tuning"),
        ("Grouping", f"binary char_wb 3-5 grams, cosine >= {config['near_duplicate_similarity']}, connected components"),
        ("Substitution", f"mapping={SUBSTITUTIONS}; eligible-word selection probability={config['edit_probability']}; "
         "one uniformly selected eligible character per selected word; ASCII alphabetic words matched by \\b[A-Za-z]+\\b"),
    ]
    write_csv(pd.DataFrame(settings, columns=["component", "setting"]), out / "table2_baseline_settings.csv")
    table3 = scores[["model", "condition", "n_messages", *METRICS]].copy()
    for name in METRICS:
        table3[name + "_percent"] = table3.pop(name) * 100
    write_csv(table3, out / "table3_validation_results_percent.csv")
    write_csv(scores, out / "validation_metrics.csv")
    write_csv(predictions, out / "validation_predictions.csv")
    statistics = pd.concat([edit_statistics(frame).assign(condition=kind)
                            for kind, frame in conditions.items() if kind != "clean"], ignore_index=True)
    write_csv(statistics, out / "perturbation_statistics.csv")
    write_csv(error_slices(predictions), out / "error_slices.csv")
    write_csv(changed_source_comparison(predictions), out / "changed_source_comparison.csv")
    pairs = paired_cases(predictions, out)
    review = write_manual_review(conditions["substitution"], out, config["manual_review_per_class"], seed)
    word = scores[scores.model == "word_baseline"].set_index("condition")
    delta = {name + "_change_percentage_points": float((word.loc["substitution", name] - word.loc["clean", name]) * 100)
             for name in METRICS}
    values = {"evaluation_partition": "validation", "test_evaluated": False,
              "table1": table1.to_dict(orient="records"), "table2": dict(settings),
              "table3": table3.to_dict(orient="records"), "metric_units_table3": "percent",
              "changes": delta, "substitution_statistics": statistics.to_dict(orient="records"),
              "data_audit": read_json(processed / "data_audit.json"), "models": model_info,
              "example": pairs.iloc[0].to_dict(), "manual_review_file": str(review.relative_to(out)),
              "manual_review_status": "Pending human review; no readability or label-preservation finding is claimed."}
    write_json(out / "report_values.json", values)
    c, e = word.loc["clean"], word.loc["substitution"]
    stats = statistics[statistics.label == "all"].iloc[0]
    example = pairs.iloc[0]
    instructions = f"""REPORT FILL-IN GUIDE (generated from actual validation predictions)

Table 1: table1_data_partitions.csv. Test counts describe the reserved partition only.
Table 2: table2_baseline_settings.csv. The baseline settings were fixed, not tuned.
Table 3: table3_validation_results_percent.csv. These values are percentages.
validation_metrics.csv and validation_predictions.csv retain unrounded proportions/probabilities.

Validation messages: {int(c.n_messages)}
Changed messages: {int(stats.n_changed)} ({stats.fraction_changed * 100:.2f}%)
Edited eligible words: {int(stats.edited_words)} / {int(stats.eligible_words)} ({stats.realized_edit_rate * 100:.2f}%)
Spam F1: clean {c.spam_f1 * 100:.2f}%, edited {e.spam_f1 * 100:.2f}%.
F1 change (edited minus clean): {delta['spam_f1_change_percentage_points']:+.2f} percentage points.
Spam recall: clean {c.spam_recall * 100:.2f}%, edited {e.spam_recall * 100:.2f}%.
Ham FPR: clean {c.ham_fpr * 100:.2f}%, edited {e.ham_fpr * 100:.2f}%.

Illustrative example (selected by clean-correct/edited-wrong, then prediction flip,
then absolute probability change; not a representative random example):
Source: {example.source_id}; true label: {example.label}
Original: {example.text_clean}
Edited: {example.text_edited}
Spam probability: {example.spam_probability_clean:.6f} -> {example.spam_probability_edited:.6f}
Prediction: {example.predicted_label_clean} -> {example.predicted_label_edited}

REQUIRES HUMAN INPUT
Open {review.relative_to(out)} and inspect each original/edited pair.
Fill readable_yes_no, label_preserved_yes_no, reviewer, and notes.
Only then state how many examples you inspected and what you observed in the report.
The pipeline does not certify that character edits preserve semantics.

NEXT EXPERIMENT
Run the compare command to change word TF-IDF to character TF-IDF (3-5 grams)
with the same split, original training messages, LR C, threshold, and paired
validation variants. Before running it, this is a planned experiment.

Scope: one development split and one perturbation seed. No test predictions were
computed. Unchanged variants remain in the main edited condition. Descriptive
changed-only and short/promotional-message slices are in error_slices.csv.
FPR is undefined in a slice with no ham; precision/F1 are zero if their denominators
are zero. Do not interpret a zero-support slice as evidence of good performance.
"""
    (out / "report_fill_in.txt").write_text(instructions, encoding="utf-8")
    baseline_figures(scores, out)


def aggregate_scores(scores: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    grouped = scores.groupby(keys, dropna=False)[METRICS].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{statistic}" for metric, statistic in grouped.columns]
    return grouped.reset_index()


def robustness_changes(scores: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ["seed", "model", "operating_point"] if c in scores]
    clean = scores[scores.condition == "clean"][[*keys, *METRICS]]
    paired = scores[scores.condition != "clean"].merge(clean, on=keys, suffixes=("", "_clean"), validate="m:1")
    for name in METRICS:
        paired[name + "_change_pp"] = 100 * (paired[name] - paired[name + "_clean"])
    paired["f1_drop_pp"] = -paired.spam_f1_change_pp
    return paired


def final_figures(summary: pd.DataFrame, out: Path) -> None:
    primary = summary[summary.operating_point == "selected_f1"]
    conditions = list(dict.fromkeys(primary.condition))
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    models = list(dict.fromkeys(primary.model))
    x, width = np.arange(len(conditions)), 0.8 / len(models)
    for i, model in enumerate(models):
        values = primary[primary.model == model].set_index("condition").loc[conditions]
        ax.bar(x + (i - (len(models) - 1) / 2) * width, values.spam_f1_mean * 100,
               width, yerr=values.spam_f1_std.fillna(0) * 100, capsize=3, label=model.replace("_", " "))
    ax.set(xticks=x, xticklabels=conditions, ylabel="Spam F1 (%)", ylim=(0, 110),
           title="Frozen test evaluation: mean and sample SD across split seeds")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False, fontsize=9)
    save_figure(fig, out / "figures" / "final_robustness")
