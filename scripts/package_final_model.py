import json
import tempfile

import mlflow
import mlflow.sklearn
import yaml

from src.data.load_data import load_data
from src.features.build_features import prepare_dataset
from src.models.factory import build_model
from src.operations.model_package import save_model_package, make_json_serializable
from src.training.train import split_data, train_and_evaluate
from src.training.tuning import run_grid_search


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def sanitize_best_params(best_params: dict) -> dict:
    return {key.split("__", 1)[-1]: value for key, value in best_params.items()}


def log_text_artifact(filename: str, content: str):
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = f"{tmp_dir}/{filename}"
        with open(path, "w", encoding="utf-8") as file:
            file.write(content)
        mlflow.log_artifact(path)


def main():
    config = load_config()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("municipios_vulnerabilidade")

    model_name = config["deployment"]["model_name"]
    reduction_name = config["deployment"]["reduction_method"]

    df = load_data(config["data"]["raw_path"])

    X, y, ids, _ = prepare_dataset(
        df=df,
        id_column=config["data"]["id_column"],
        target_source_column=config["data"]["target_source_column"],
        target_column=config["data"]["target_column"]
    )

    X_train, X_test, y_train, y_test, _, _ = split_data(
        X,
        y,
        ids,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"]
    )

    base_model = build_model(
        config=config,
        model_name=model_name,
        reduction_name=reduction_name
    )
    final_model = base_model
    best_params = {}

    if config["search"]["run_grid_search"] and model_name in ["decision_tree", "random_forest"]:
        grid = run_grid_search(
            model=base_model,
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            config=config
        )
        final_model = grid.best_estimator_
        best_params = sanitize_best_params(grid.best_params_)

    trained_model, metrics = train_and_evaluate(
        model=final_model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

    metadata = {
        "package_name": config["deployment"]["package_name"],
        "model_name": model_name,
        "reduction_method": reduction_name,
        "model_version": config["deployment"]["model_version"],
        "id_column": config["data"]["id_column"],
        "target_column": config["data"]["target_column"],
        "test_metrics": metrics,
        "best_params": best_params
    }

    serializable_metadata = make_json_serializable(metadata)

    package_dir = save_model_package(
        trained_model=trained_model,
        feature_columns=list(X_train.columns),
        metadata=serializable_metadata,
        reference_features=X_train,
        package_root=config["deployment"]["package_dir"],
        package_name=config["deployment"]["package_name"],
        model_name=model_name,
        reduction_name=reduction_name,
        version=config["deployment"]["model_version"]
    )

    with mlflow.start_run(run_name=f"deploy_{model_name}_{reduction_name}"):
        mlflow.set_tag("stage", "parte_6_operacionalizacao")
        mlflow.log_param("deployment_model", model_name)
        mlflow.log_param("deployment_reduction", reduction_name)
        mlflow.log_param("deployment_version", config["deployment"]["model_version"])

        for metric_name in ["accuracy", "precision", "recall", "f1"]:
            mlflow.log_metric(metric_name, metrics[metric_name])

        log_text_artifact(
            "deployment_metadata.json",
            json.dumps(serializable_metadata, ensure_ascii=False, indent=2)
        )
        mlflow.log_artifacts(package_dir, artifact_path="inference_package")
        mlflow.sklearn.log_model(trained_model, artifact_path="deployed_model")

    print(f"Pacote salvo em: {package_dir}")


if __name__ == "__main__":
    main()