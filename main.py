import os
import tempfile

import mlflow
import mlflow.sklearn
import yaml

from src.data.load_data import load_data
from src.data.diagnose_data import diagnose_dataset, print_diagnosis
from src.features.build_features import prepare_dataset
from src.models.perceptron import build_perceptron_pipeline
from src.models.decision_tree import build_decision_tree_pipeline
from src.models.random_forest import build_random_forest_pipeline
from src.training.train import split_data, train_and_evaluate
from src.training.tuning import run_grid_search


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_model(config: dict, model_name: str):
    if model_name == "perceptron":
        params = config["perceptron"]
        return build_perceptron_pipeline(**params)

    if model_name == "decision_tree":
        params = config["decision_tree"]
        return build_decision_tree_pipeline(**params)

    if model_name == "random_forest":
        params = config["random_forest"]
        return build_random_forest_pipeline(**params)

    raise ValueError(f"Modelo não suportado: {model_name}")


def get_model_params(config: dict, model_name: str) -> dict:
    if model_name not in config:
        return {}
    return config[model_name]


def log_text_artifact(filename: str, content: str):
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, filename)
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
        mlflow.log_artifact(file_path)


def sanitize_best_params(best_params: dict) -> dict:

    clean_params = {}
    for key, value in best_params.items():
        clean_key = key.split("__", 1)[-1]
        clean_params[clean_key] = value
    return clean_params


def run_experiment(config: dict, model_name: str, X_train, X_test, y_train, y_test):
    base_model = build_model(config, model_name)
    model_params = get_model_params(config, model_name)

    with mlflow.start_run(run_name=model_name):
        mlflow.set_tag("project", "municipios_vulnerabilidade")
        mlflow.set_tag("stage", "parte_3_experimentacao_sistematica")
        mlflow.set_tag("model_name", model_name)

        mlflow.log_param("selected_model", model_name)
        mlflow.log_param("test_size", config["split"]["test_size"])
        mlflow.log_param("random_state", config["split"]["random_state"])
        mlflow.log_param("cv_folds", config["cross_validation"]["cv_folds"])
        mlflow.log_param("scoring", config["cross_validation"]["scoring"])
        mlflow.log_param("run_grid_search", config["search"]["run_grid_search"])

        for param_name, param_value in model_params.items():
            mlflow.log_param(param_name, param_value)

        final_model = base_model

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

            mlflow.log_metric("cv_best_score", grid.best_score_)

            for param_name, param_value in best_params.items():
                mlflow.log_param(f"best_{param_name}", param_value)

            log_text_artifact("best_params.txt", str(best_params))

        trained_model, metrics = train_and_evaluate(
            model=final_model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test
        )

        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1", metrics["f1"])

        log_text_artifact("classification_report.txt", metrics["classification_report"])
        log_text_artifact("confusion_matrix.txt", str(metrics["confusion_matrix"]))

        mlflow.sklearn.log_model(
            sk_model=trained_model,
            name="model"
        )

        print(f"\n=== EXPERIMENTO: {model_name} ===")
        print(f"Shape treino: {X_train.shape}")
        print(f"Shape teste: {X_test.shape}")

        if config["search"]["run_grid_search"] and model_name in ["decision_tree", "random_forest"]:
            print(f"Melhor score médio em CV: {grid.best_score_:.4f}")
            print("Melhores hiperparâmetros:")
            for key, value in best_params.items():
                print(f"{key}: {value}")

        print("\nMétricas de teste:")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-score:  {metrics['f1']:.4f}")

        print("\nClassification Report:")
        print(metrics["classification_report"])

        print("\nConfusion Matrix:")
        print(metrics["confusion_matrix"])


def main():
    config = load_config()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("municipios_vulnerabilidade")

    df = load_data(config["data"]["raw_path"])

    diagnosis = diagnose_dataset(df)
    print_diagnosis(diagnosis)

    X, y, ids, df_prepared = prepare_dataset(
        df=df,
        id_column=config["data"]["id_column"],
        target_source_column=config["data"]["target_source_column"],
        target_column=config["data"]["target_column"]
    )

    X_train, X_test, y_train, y_test, ids_train, ids_test = split_data(
        X,
        y,
        ids,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"]
    )

    selected_model = config["models"]["selected_model"]

    if selected_model == "all":
        model_names = ["perceptron", "decision_tree", "random_forest"]
    else:
        model_names = [selected_model]

    for model_name in model_names:
        run_experiment(
            config=config,
            model_name=model_name,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test
        )


if __name__ == "__main__":
    main()