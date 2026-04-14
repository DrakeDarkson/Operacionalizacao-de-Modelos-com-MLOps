import mlflow


def start_experiment(experiment_name: str):
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run()


def log_params(params: dict):
    for key, value in params.items():
        mlflow.log_param(key, value)


def log_metrics(metrics: dict):
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(key, value)


def log_model(model, model_name: str):
    mlflow.sklearn.log_model(model, model_name)