import pandas as pd
from scipy.stats import ks_2samp


def detect_data_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    alpha: float = 0.05
) -> dict:
    common_columns = [col for col in reference_df.columns if col in current_df.columns]
    report = {
        "alpha": alpha,
        "drifted_features": [],
        "feature_results": {}
    }

    for col in common_columns:
        ref = pd.to_numeric(reference_df[col], errors="coerce").dropna()
        cur = pd.to_numeric(current_df[col], errors="coerce").dropna()

        if len(ref) == 0 or len(cur) == 0:
            continue

        statistic, p_value = ks_2samp(ref, cur)
        drift_flag = p_value < alpha

        report["feature_results"][col] = {
            "ks_statistic": float(statistic),
            "p_value": float(p_value),
            "drift_detected": bool(drift_flag)
        }

        if drift_flag:
            report["drifted_features"].append(col)

    report["drifted_features_count"] = len(report["drifted_features"])
    return report


def detect_model_drift(
    baseline_metrics: dict,
    current_metrics: dict,
    threshold_drop: float = 0.03
) -> dict:
    baseline_f1 = baseline_metrics["f1"]
    current_f1 = current_metrics["f1"]
    baseline_recall = baseline_metrics["recall"]
    current_recall = current_metrics["recall"]

    f1_drop = baseline_f1 - current_f1
    recall_drop = baseline_recall - current_recall

    return {
        "baseline_f1": float(baseline_f1),
        "current_f1": float(current_f1),
        "f1_drop": float(f1_drop),
        "baseline_recall": float(baseline_recall),
        "current_recall": float(current_recall),
        "recall_drop": float(recall_drop),
        "model_drift_detected": bool(f1_drop > threshold_drop or recall_drop > threshold_drop)
    }