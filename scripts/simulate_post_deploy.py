import json
import os
import tempfile

import mlflow
import pandas as pd
import yaml

from src.data.load_data import load_data
from src.features.build_features import prepare_dataset
from src.monitoring.drift import detect_data_drift, detect_model_drift
from src.monitoring.metrics import compute_business_metrics, compute_technical_metrics
from src.operations.model_package import find_latest_package, load_model_package
from src.training.train import split_data


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def log_json_artifact(filename: str, payload: dict):
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, filename)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        mlflow.log_artifact(path)


def main():
    config = load_config()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("municipios_vulnerabilidade")

    package_dir = find_latest_package(
        package_root=config["deployment"]["package_dir"],
        package_name=config["deployment"]["package_name"],
        model_name=config["deployment"]["model_name"],
        reduction_name=config["deployment"]["reduction_method"]
    )

    model, metadata, feature_columns, reference_features = load_model_package(package_dir)

    df = load_data(config["data"]["raw_path"])
    X, y, ids, _ = prepare_dataset(
        df=df,
        id_column=config["data"]["id_column"],
        target_source_column=config["data"]["target_source_column"],
        target_column=config["data"]["target_column"]
    )

    _, X_current, _, y_current, _, _ = split_data(
        X,
        y,
        ids,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"]
    )

    X_current = X_current[feature_columns].copy()

    y_pred = model.predict(X_current)
    technical_metrics = compute_technical_metrics(y_current, y_pred)
    business_metrics = compute_business_metrics(
        y_true=y_current.tolist(),
        y_pred=y_pred.tolist(),
        positive_class=config["business_metrics"]["positive_class"]
    )

    data_drift_report = detect_data_drift(
        reference_df=reference_features,
        current_df=X_current,
        alpha=config["monitoring"]["drift_alpha"]
    )

    model_drift_report = detect_model_drift(
        baseline_metrics=metadata["test_metrics"],
        current_metrics=technical_metrics,
        threshold_drop=config["monitoring"]["performance_drop_threshold"]
    )

    with mlflow.start_run(run_name=f"monitor_{metadata['model_name']}_{metadata['reduction_method']}"):
        mlflow.set_tag("stage", "parte_6_monitoramento")
        mlflow.log_param("model_name", metadata["model_name"])
        mlflow.log_param("model_version", metadata["model_version"])
        mlflow.log_param("reduction_method", metadata["reduction_method"])

        for key, value in technical_metrics.items():
            if key != "confusion_matrix":
                mlflow.log_metric(f"post_deploy_{key}", value)

        for key, value in business_metrics.items():
            mlflow.log_metric(f"business_{key}", value)

        mlflow.log_metric("drifted_features_count", data_drift_report["drifted_features_count"])
        mlflow.log_metric("model_drift_detected", int(model_drift_report["model_drift_detected"]))

        log_json_artifact("technical_metrics.json", technical_metrics)
        log_json_artifact("business_metrics.json", business_metrics)
        log_json_artifact("data_drift_report.json", data_drift_report)
        log_json_artifact("model_drift_report.json", model_drift_report)

    print("Monitoramento pós-deploy concluído.")
    print(json.dumps(technical_metrics, ensure_ascii=False, indent=2))
    print(json.dumps(business_metrics, ensure_ascii=False, indent=2))
    print(json.dumps(data_drift_report, ensure_ascii=False, indent=2))
    print(json.dumps(model_drift_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()