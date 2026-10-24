from __future__ import annotations

from collections.abc import Sequence
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = (
    "d531_probability",
    "d532_hls_probability",
    "d532_probability",
    "d533_probability",
    "d534_selected_probability",
    "hls_obs_count",
    "hls_ndvi_mean",
    "hls_ndvi_max",
    "hls_ndvi_min",
    "hls_ndwi_mean",
    "hls_blue_mean",
    "hls_green_mean",
    "hls_red_mean",
    "hls_nir_mean",
    "hls_swir_obs_count",
    "hls_ndmi_mean",
    "hls_swir1_mean",
    "hls_swir2_mean",
)

IDENTITY_COLUMNS = ("pair_id", "endpoint_id", "endpoint_role", "expected_label")


def binary_labels(values: pd.Series) -> pd.Series:
    labels = pd.to_numeric(values, errors="raise")
    numeric = labels.to_numpy(dtype=float)
    if labels.isna().any() or not np.isfinite(numeric).all() or not labels.isin([0, 1]).all():
        raise ValueError("expected_label must contain only 0 and 1")
    return labels.astype(int)


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=9801,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def validate_endpoint_table(frame: pd.DataFrame, expected_pairs: int | None = None) -> dict[str, int]:
    required = set(IDENTITY_COLUMNS + FEATURE_COLUMNS)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    if frame["endpoint_id"].isna().any() or frame["pair_id"].isna().any():
        raise ValueError("pair_id and endpoint_id must not be missing")
    if frame["endpoint_id"].duplicated().any():
        raise ValueError("endpoint_id must be unique")
    labels = binary_labels(frame["expected_label"])
    grouped = frame.assign(expected_label=labels).groupby("pair_id", sort=False)
    invalid = []
    for pair_id, rows in grouped:
        if (
            len(rows) != 2
            or rows["endpoint_id"].nunique() != 2
            or set(rows["endpoint_role"].astype(str)) != {"A", "B"}
            or set(rows["expected_label"]) != {0, 1}
        ):
            invalid.append(str(pair_id))
    if invalid:
        raise ValueError(f"invalid strict pairs: {', '.join(invalid[:10])}")
    pair_count = grouped.ngroups
    if expected_pairs is not None and pair_count != expected_pairs:
        raise ValueError(f"expected {expected_pairs} pairs, found {pair_count}")
    numeric = frame.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("feature matrix contains infinite values")
    all_missing = [name for name in FEATURE_COLUMNS if numeric[name].isna().all()]
    if all_missing:
        raise ValueError(f"all-missing feature columns: {', '.join(all_missing)}")
    return {
        "pairs": pair_count,
        "endpoints": len(frame),
        "features": len(FEATURE_COLUMNS),
        "missing_values": int(numeric.isna().sum().sum()),
    }


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")


def fit_pipeline(frame: pd.DataFrame, expected_pairs: int | None = None) -> tuple[Pipeline, dict[str, int]]:
    audit = validate_endpoint_table(frame, expected_pairs)
    labels = binary_labels(frame["expected_label"]).to_numpy()
    model = build_pipeline()
    model.fit(feature_matrix(frame), labels)
    return model, audit


def predict_probabilities(model: Pipeline, frame: pd.DataFrame, expected_pairs: int | None = None) -> np.ndarray:
    validate_endpoint_table(frame, expected_pairs)
    probabilities = model.predict_proba(feature_matrix(frame))[:, 1]
    if probabilities.shape != (len(frame),) or not np.isfinite(probabilities).all():
        raise ValueError("model returned invalid probabilities")
    return probabilities


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("invalid binomial counts")
    probability = successes / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    margin = z * np.sqrt(probability * (1 - probability) / total + z * z / (4 * total * total)) / denominator
    return float(center - margin), float(center + margin)


def endpoint_metrics(labels: Sequence[int], probabilities: Sequence[float], threshold: float = 0.5) -> dict[str, float | int]:
    truth = np.asarray(labels, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if truth.shape != scores.shape or truth.ndim != 1 or len(truth) == 0:
        raise ValueError("labels and probabilities must be nonempty one-dimensional arrays with equal length")
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
    total = int(len(truth))
    correct = int((truth == predictions).sum())
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    low, high = wilson_interval(correct, total)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "wilson95_low": low,
        "wilson95_high": high,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold,
    }


def pair_results(frame: pd.DataFrame, probabilities: Sequence[float]) -> pd.DataFrame:
    scores = np.asarray(probabilities, dtype=float)
    if scores.shape != (len(frame),):
        raise ValueError("probability count does not match endpoint count")
    rows = frame.loc[:, IDENTITY_COLUMNS].copy()
    rows["expected_label"] = binary_labels(rows["expected_label"])
    rows["probability"] = scores
    output = []
    for pair_id, pair in rows.groupby("pair_id", sort=False):
        negative = pair.loc[pair["expected_label"] == 0].iloc[0]
        positive = pair.loc[pair["expected_label"] == 1].iloc[0]
        difference = float(positive["probability"] - negative["probability"])
        output.append(
            {
                "pair_id": pair_id,
                "negative_endpoint_id": negative["endpoint_id"],
                "positive_endpoint_id": positive["endpoint_id"],
                "pair_correct": difference > 0.0,
                "tie": difference == 0.0,
            }
        )
    return pd.DataFrame(output)


def pair_metrics(results: pd.DataFrame) -> dict[str, float | int]:
    total = int(len(results))
    if total == 0:
        raise ValueError("pair results are empty")
    correct = int(results["pair_correct"].astype(bool).sum())
    ties = int(results["tie"].astype(bool).sum())
    low, high = wilson_interval(correct, total)
    return {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "ties": ties,
        "accuracy": correct / total,
        "wilson95_low": low,
        "wilson95_high": high,
        "comparator": "strict_greater_than",
        "tie_rule": "failure",
    }
