from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.models.perceptron import build_perceptron_pipeline
from src.models.decision_tree import build_decision_tree_pipeline
from src.models.random_forest import build_random_forest_pipeline


def get_reduction_object(config: dict, reduction_name: str):
    if reduction_name == "none":
        return None

    if reduction_name == "pca":
        return PCA(n_components=config["pca"]["n_components"])

    if reduction_name == "lda":
        return LinearDiscriminantAnalysis()

    raise ValueError(f"Técnica de redução não suportada: {reduction_name}")


def build_model(config: dict, model_name: str, reduction_name: str = "none"):
    reduction = get_reduction_object(config, reduction_name)

    if model_name == "perceptron":
        params = config["perceptron"].copy()
        return build_perceptron_pipeline(reduction=reduction, **params)

    if model_name == "decision_tree":
        params = config["decision_tree"].copy()
        return build_decision_tree_pipeline(reduction=reduction, **params)

    if model_name == "random_forest":
        params = config["random_forest"].copy()
        return build_random_forest_pipeline(reduction=reduction, **params)

    raise ValueError(f"Modelo não suportado: {model_name}")


def get_model_params(config: dict, model_name: str) -> dict:
    return config.get(model_name, {})