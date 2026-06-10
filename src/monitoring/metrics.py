from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

from src.monitoring.config import (
    BASELINE_METRICS_PATH,
    KEY_FEATURES,
    MAE_INCREASE_RATIO,
    MODEL_PATH,
    PSI_THRESHOLD,
    R2_DROP_THRESHOLD,
)

LOG_OFFSET = 50


def add_model_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Add model predictions for Evidently regression reports."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if "target" not in df.columns:
        raise ValueError("Column 'target' is required")

    feature_cols = [col for col in df.columns if col not in {"timestamp", "target", "prediction"}]
    x = df[feature_cols].copy()

    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))
    model_features = model.get_booster().feature_names

    for col in model_features:
        if col not in x.columns:
            x[col] = 0.0

    preds_log = model.predict(x[model_features].astype(np.float64))
    preds = np.expm1(preds_log) - LOG_OFFSET

    out = df.copy()
    out["prediction"] = preds
    return out


def calculate_psi(reference: pd.Series, current: pd.Series, buckets: int = 10) -> float:
    ref = reference.dropna().astype(float)
    cur = current.dropna().astype(float)
    if ref.empty or cur.empty:
        return 0.0

    quantiles = np.linspace(0, 1, buckets + 1)
    breaks = np.unique(np.quantile(ref, quantiles))
    if len(breaks) < 2:
        return 0.0

    breaks[0] = -np.inf
    breaks[-1] = np.inf

    ref_counts = np.histogram(ref, bins=breaks)[0]
    cur_counts = np.histogram(cur, bins=breaks)[0]

    ref_perc = np.where(ref_counts == 0, 1e-6, ref_counts / ref_counts.sum())
    cur_perc = np.where(cur_counts == 0, 1e-6, cur_counts / cur_counts.sum())

    return float(np.sum((cur_perc - ref_perc) * np.log(cur_perc / ref_perc)))


def compute_psi_by_feature(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
) -> dict[str, float]:
    psi_scores: dict[str, float] = {}
    for feature in features:
        if feature not in reference.columns or feature not in current.columns:
            continue
        psi_scores[feature] = calculate_psi(reference[feature], current[feature])
    return psi_scores


def load_baseline_metrics() -> dict[str, float]:
    if BASELINE_METRICS_PATH.exists():
        return json.loads(BASELINE_METRICS_PATH.read_text(encoding="utf-8"))
    return {"mae": 16.6152, "r2": 0.7552}


def save_baseline_metrics(mae: float, r2: float) -> None:
    BASELINE_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_METRICS_PATH.write_text(
        json.dumps({"mae": mae, "r2": r2}, indent=2),
        encoding="utf-8",
    )


def evaluate_current_model(
    df: pd.DataFrame,
    split_ratio: float = 0.8,
) -> dict[str, float]:
    if not MODEL_PATH.exists():
        return {}

    split_idx = int(len(df) * split_ratio)
    test_df = df.iloc[split_idx:].copy()
    if test_df.empty or "target" not in test_df.columns:
        return {}

    feature_cols = [col for col in df.columns if col not in {"timestamp", "target"}]
    x_test = test_df[feature_cols].copy()
    y_test = test_df["target"]

    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))
    model_features = model.get_booster().feature_names

    for col in model_features:
        if col not in x_test.columns:
            x_test[col] = 0.0

    preds_log = model.predict(x_test[model_features].astype(np.float64))
    preds = np.expm1(preds_log) - LOG_OFFSET

    return {
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
    }


def evaluate_retrain_trigger(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    current_metrics: dict[str, float] | None = None,
) -> dict:
    psi_scores = compute_psi_by_feature(reference, current, KEY_FEATURES)
    max_psi = max(psi_scores.values()) if psi_scores else 0.0
    max_psi_feature = max(psi_scores, key=psi_scores.get) if psi_scores else None
    feature_psi_scores = {feature: score for feature, score in psi_scores.items() if feature != "target"}
    drifted_features = sorted(
        [feature for feature, score in feature_psi_scores.items() if score > PSI_THRESHOLD],
        key=lambda feature: feature_psi_scores[feature],
        reverse=True,
    )
    target_psi = psi_scores.get("target")

    baseline = load_baseline_metrics()
    current_metrics = current_metrics or {}
    current_mae = current_metrics.get("mae")
    current_r2 = current_metrics.get("r2")

    drift_reasons: list[str] = []
    quality_reasons: list[str] = []
    if drifted_features:
        top_feature = drifted_features[0]
        drift_reasons.append(f"PSI {feature_psi_scores[top_feature]:.3f} on '{top_feature}' > {PSI_THRESHOLD}")
        if len(drifted_features) > 1:
            drift_reasons.append(f"Drift also detected in {len(drifted_features) - 1} more feature(s)")
    if target_psi is not None and target_psi > PSI_THRESHOLD:
        drift_reasons.append(f"Target PSI {target_psi:.3f} > {PSI_THRESHOLD}")

    if current_mae is not None and baseline.get("mae"):
        mae_increase = (current_mae - baseline["mae"]) / baseline["mae"]
        if mae_increase > MAE_INCREASE_RATIO:
            quality_reasons.append(
                f"MAE increased {mae_increase:.1%} vs baseline " f"({current_mae:.2f} vs {baseline['mae']:.2f})"
            )
    else:
        mae_increase = None

    if current_r2 is not None and baseline.get("r2") is not None:
        r2_drop = baseline["r2"] - current_r2
        if r2_drop > R2_DROP_THRESHOLD:
            quality_reasons.append(
                f"R2 dropped {r2_drop:.3f} vs baseline " f"({current_r2:.3f} vs {baseline['r2']:.3f})"
            )
    else:
        r2_drop = None

    quality_degraded = bool(quality_reasons)
    drift_detected = bool(drift_reasons)
    if quality_degraded:
        alert_level = "critical"
        reasons = quality_reasons + drift_reasons
    elif drift_detected:
        alert_level = "warning"
        reasons = drift_reasons
    else:
        alert_level = "none"
        reasons = []

    return {
        "should_retrain": quality_degraded,
        "alert_level": alert_level,
        "reasons": reasons,
        "drift_reasons": drift_reasons,
        "quality_reasons": quality_reasons,
        "drift_detected": drift_detected,
        "quality_degraded": quality_degraded,
        "drifted_features": drifted_features,
        "psi_scores": psi_scores,
        "max_psi": max_psi,
        "max_psi_feature": max_psi_feature,
        "baseline_metrics": baseline,
        "current_metrics": current_metrics,
        "mae_increase_ratio": mae_increase,
        "r2_drop": r2_drop,
    }
