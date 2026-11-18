"""Protect experimental validity, then exercise every stage on synthetic fixtures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sms_spam.bootstrap import paired_bootstrap
from sms_spam.common import read_csv, read_json
from sms_spam.data import (assert_disjoint, group_holdout, load_development,
                           load_raw, near_duplicate_groups, prepare, remove_exact_duplicates)
from sms_spam.experiments import final_evaluation, run_midsemester, tune
from sms_spam.model import low_fpr_threshold, metrics, select_threshold, training_features
from sms_spam.perturb import make_augmentation, make_condition, perturb_text


@pytest.fixture
def config():
    value = read_json(Path(__file__).resolve().parents[1] / "config.json")
    value.update(split_seeds=[42, 43], split_candidates=40, near_duplicate_similarity=0.99,
                 min_df=1, C_candidates=[0.1, 1.0], threshold_candidates=[0.3, 0.5],
                 manual_review_per_class=2, bootstrap_replicates=40)
    return value


def sample_frame():
    return pd.DataFrame({"source_id": ["one", "two"], "group_id": ["g1", "g2"],
                         "label": ["ham", "spam"], "y": [0, 1],
                         "text": ["Please meet me for dinner.", "FREE cash prize, claim today!"]})


def make_synthetic_dataset(root: Path):
    rng = np.random.default_rng(123)
    rows = []
    for i in range(160):
        label = "spam" if i % 2 else "ham"
        prefix = "free cash prize claim reward" if label == "spam" else "dinner friend lunch meeting tomorrow"
        words = ["".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"), size=8)) for _ in range(8)]
        rows.append(f"{label}\t{prefix} {' '.join(words)} {i}\n")
    target = root / "data" / "raw" / "SMSSpamCollection"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(rows), encoding="utf-8")


def test_raw_parser_preserves_sms_and_unicode_line_separator(tmp_path):
    path = tmp_path / "raw"
    path.write_bytes("ham\t NA\u0085hello \r\nspam\tfree\tclaim\n".encode())
    frame = load_raw(path)
    assert frame.text.tolist() == [" NA\u0085hello ", "free\tclaim"]
    assert frame.y.tolist() == [0, 1]


def test_exact_duplicate_and_conflicting_label_policy():
    frame = pd.DataFrame({"text": ["hello", "hello", "free", "free", "Hello"], "y": [0, 0, 0, 1, 0]})
    unique, duplicates, conflicts = remove_exact_duplicates(frame)
    assert unique.text.tolist() == ["hello", "Hello"]
    assert len(duplicates) == 1 and len(conflicts) == 2


def test_normalized_duplicates_share_group_and_groups_do_not_cross_split():
    frame = pd.DataFrame({"source_id": [f"s{i}" for i in range(12)],
                          "text": ["HELLO friend", "hello   friend", "Free new prize", "free NEW prize",
                                   "Lunch with mom", "Buy fresh apples", "See you tomorrow", "Laundry is done",
                                   "Cash reward now", "Claim the award", "Winner call today", "Urgent offer"],
                          "y": [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]})
    frame["group_id"], edges = near_duplicate_groups(frame, 0.9)
    assert frame.group_id.iat[0] == frame.group_id.iat[1]
    assert frame.group_id.iat[2] == frame.group_id.iat[3]
    development, held = group_holdout(frame, 0.25, 42, 100)
    assert_disjoint(development, held)
    assert len(edges) >= 2


def test_perturbations_preserve_labels_and_are_independent_of_row_order():
    frame = sample_frame()
    for kind in ["substitution", "space", "swap"]:
        first = make_condition(frame, kind, 0.2, 31415).set_index("source_id")
        reordered = make_condition(frame.iloc[::-1], kind, 0.2, 31415).set_index("source_id")
        assert first.text.to_dict() == reordered.text.to_dict()
        assert first.y.tolist() == frame.y.tolist()
        different_labels = frame.copy()
        different_labels["y"] = 1 - frame.y
        assert make_condition(different_labels, kind, 0.2, 31415).text.tolist() == first.text.tolist()


def test_zero_full_and_nested_edit_rates():
    text = "Free cash prize, please claim tomorrow! 123 xyz."
    for kind in ["substitution", "space", "swap"]:
        zero = perturb_text(text, kind, 0, 7, "a")
        full = perturb_text(text, kind, 1, 7, "a")
        assert zero["text"] == text and zero["edited_words"] == 0
        assert full["edited_words"] == full["eligible_words"] > 0
        assert full["text"] != text
    low = perturb_text(text, "substitution", 0.1, 7, "a")["text"].split()
    high = perturb_text(text, "substitution", 0.3, 7, "a")["text"].split()
    assert all(a == original or a == b for original, a, b in zip(text.split(), low, high))
    ineligible = perturb_text("123 xyz", "substitution", 1, 7, "a")
    assert ineligible["eligible_words"] == 0 and not ineligible["changed"]


def test_metric_denominators_and_threshold_boundary():
    result = metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert result["spam_f1"] == result["ham_fpr"] == 0.5
    assert [result[k] for k in ["tn", "fp", "fn", "tp"]] == [1, 1, 1, 1]
    threshold, score = select_threshold([1, 0], [0.5, 0.1], [0.5, 0.6])
    assert threshold == 0.5 and score["spam_f1"] == 1
    p = np.array([0.1, 0.2, 0.2, 0.9, 0.9, 0.95])
    y = np.array([0, 0, 0, 0, 0, 1])
    threshold = low_fpr_threshold(y, p, 0.2)
    assert (p[y == 0] >= threshold).mean() <= 0.2
    assert threshold > 0.9  # tied ham scores must both fall below the cutoff


def test_training_only_vocabulary_and_source_weights(config):
    train = sample_frame()
    augmentation = make_augmentation(train, 0.2, 10042)
    assert len(augmentation) == len(train)
    assert set(augmentation.condition) <= {"substitution", "space"}
    vectorizer, matrix, y, weights = training_features(train, "word", config, augmentation)
    assert weights.sum() == len(train)
    assert len(y) == 2 * len(train)
    assert np.all(weights == 0.5)
    before = dict(vectorizer.vocabulary_)
    vectorizer.transform(["validationonlytoken neverseenword"])
    assert vectorizer.vocabulary_ == before
    assert "validationonlytoken" not in vectorizer.vocabulary_


def test_bootstrap_pairing_identity_and_alignment():
    rows = []
    for model in ["word_baseline", "character_baseline", "word_augmented", "character_augmented"]:
        for seed in [1, 2]:
            for condition in ["clean", "substitution_p0.2"]:
                for i in range(12):
                    rows.append({"source_id": f"s{i}", "group_id": f"g{i // 2}", "model": model,
                                 "seed": seed, "condition": condition, "operating_point": "selected_f1",
                                 "y": i % 2, "predicted_y": int(i % 3 == 0)})
    frame = pd.DataFrame(rows).sample(frac=1, random_state=99)
    for unit in ["message", "group"]:
        intervals = paired_bootstrap(frame, 100, 4, unit)
        assert np.allclose(intervals[["difference_pp", "ci95_low_pp", "ci95_high_pp"]], 0)
    with pytest.raises(ValueError, match="paired source"):
        paired_bootstrap(frame.iloc[1:], 10, 4)


def test_bootstrap_known_difference_uses_metrics_not_mean_probability():
    rows = []
    for model in ["word_baseline", "character_baseline"]:
        for seed in [1, 2]:
            for condition in ["clean", "substitution_p0.2"]:
                for i in range(20):
                    y = i % 2
                    rows.append({"source_id": f"s{i}", "group_id": f"g{i}", "model": model,
                                 "seed": seed, "condition": condition, "operating_point": "selected_f1",
                                 "y": y, "predicted_y": y if model == "character_baseline" else 1 - y})
    intervals = paired_bootstrap(pd.DataFrame(rows), 100, 11)
    assert (intervals[intervals.metric == "spam_f1"].difference_pp == 100).all()
    assert (intervals[intervals.metric == "ham_fpr"].difference_pp == -100).all()
    assert (intervals[intervals.metric == "f1_drop"].difference_pp == 0).all()


def test_all_stages_on_synthetic_fixture_keep_real_test_untouched(tmp_path, config):
    make_synthetic_dataset(tmp_path)
    manifest = prepare(tmp_path, config)
    assert manifest["audit"]["raw_messages"] == 160
    reserved_before = (tmp_path / "data/processed/test_reserved.csv").read_bytes()
    test = read_csv(tmp_path / "data/processed/test_reserved.csv")
    for seed in config["split_seeds"]:
        train, validation = load_development(tmp_path, seed)
        assert_disjoint(train, validation, test)
        assert set(train.y) == set(validation.y) == set(test.y) == {0, 1}
    out = run_midsemester(tmp_path, config)
    values = read_json(out / "report_values.json")
    assert values["test_evaluated"] is False
    assert (out / "figures/baseline_validation_metrics.png").exists()
    review = out / values["manual_review_file"]
    original_review = review.read_bytes()
    review.write_bytes(original_review + b"\n")
    run_midsemester(tmp_path, config)
    assert review.read_bytes() == original_review + b"\n"
    compare = run_midsemester(tmp_path, config, compare=True)
    assert set(read_csv(compare / "validation_metrics.csv").model) == {"word_baseline", "character_baseline"}
    frozen_dir = tune(tmp_path, config, low_fpr=True, rates=[0.1, 0.2, 0.3])
    frozen = read_json(frozen_dir / "frozen_manifest.json")
    assert len(frozen["models"]) == 8
    assert not frozen["test_predictions_computed"]
    assert not (frozen_dir / "final_test").exists()
    assert set(read_csv(frozen_dir / "validation_predictions.csv").condition) == {"clean", "substitution", "space"}
    assert (tmp_path / "data/processed/test_reserved.csv").read_bytes() == reserved_before
    result = final_evaluation(tmp_path, "frozen_development")
    final_predictions = read_csv(result / "test_predictions.csv")
    assert set(final_predictions.source_id) == set(test.source_id)
    assert "swap_p0.2" in set(final_predictions.condition)
    assert len(read_csv(result / "test_metrics_by_seed.csv")) == 2 * 4 * 10 * 2
    assert (result / "paired_bootstrap_contrasts.csv").exists()
    assert (tmp_path / "data/processed/test_reserved.csv").read_bytes() == reserved_before
    # A modified test file must be rejected, never accepted as the frozen holdout.
    (tmp_path / "data/processed/test_reserved.csv").write_bytes(reserved_before + b"\n")
    with pytest.raises(ValueError, match="Reserved test file"):
        final_evaluation(tmp_path, "frozen_development")


def test_changed_preparation_settings_are_rejected(tmp_path, config):
    make_synthetic_dataset(tmp_path)
    prepare(tmp_path, config)
    changed = dict(config, test_seed=1234)
    with pytest.raises(ValueError, match="settings changed"):
        prepare(tmp_path, changed)
