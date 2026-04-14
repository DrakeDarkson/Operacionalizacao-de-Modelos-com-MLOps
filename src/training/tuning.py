from sklearn.model_selection import GridSearchCV


def prefix_param_grid(model_name: str, param_grid: dict) -> dict:

    if model_name == "decision_tree":
        prefix = "tree__"
    elif model_name == "random_forest":
        prefix = "rf__"
    elif model_name == "perceptron":
        prefix = "perceptron__"
    else:
        raise ValueError(f"Modelo não suportado para tuning: {model_name}")

    return {f"{prefix}{key}": value for key, value in param_grid.items()}


def run_grid_search(model, model_name: str, X_train, y_train, config: dict):

    param_grid = config["grid_search"].get(model_name)

    if not param_grid:
        raise ValueError(f"Não há grade de hiperparâmetros para o modelo: {model_name}")

    prefixed_grid = prefix_param_grid(model_name, param_grid)

    grid = GridSearchCV(
        estimator=model,
        param_grid=prefixed_grid,
        cv=config["cross_validation"]["cv_folds"],
        scoring=config["cross_validation"]["scoring"],
        n_jobs=config["cross_validation"]["n_jobs"],
        refit=True
    )

    grid.fit(X_train, y_train)
    return grid