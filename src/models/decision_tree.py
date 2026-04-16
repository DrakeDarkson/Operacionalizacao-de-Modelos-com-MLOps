from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


def build_decision_tree_pipeline(
    criterion: str = "gini",
    max_depth=None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    ccp_alpha: float = 0.0,
    random_state: int = 42,
    reduction=None
) -> Pipeline:

    steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if reduction is not None:
        steps.append(("reduction", reduction))

    steps.append((
        "tree",
        DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
            random_state=random_state
        )
    ))

    return Pipeline(steps)