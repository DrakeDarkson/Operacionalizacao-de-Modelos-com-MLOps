from sklearn.impute import SimpleImputer
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_perceptron_pipeline(
    max_iter: int = 1000,
    tol: float = 1e-3,
    random_state: int = 42,
    reduction=None
) -> Pipeline:

    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]

    if reduction is not None:
        steps.append(("reduction", reduction))

    steps.append((
        "perceptron",
        Perceptron(
            max_iter=max_iter,
            tol=tol,
            random_state=random_state
        )
    ))

    return Pipeline(steps)