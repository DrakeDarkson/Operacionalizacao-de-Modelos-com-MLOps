from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def build_random_forest_pipeline(
    n_estimators: int = 200,
    max_depth=10,
    min_samples_split: int = 10,
    min_samples_leaf: int = 2,
    max_features: str = "sqrt",
    random_state: int = 42,
    n_jobs: int = -1,
    reduction=None
) -> Pipeline:

    steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if reduction is not None:
        steps.append(("reduction", reduction))

    steps.append((
        "rf",
        RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs
        )
    ))

    return Pipeline(steps)