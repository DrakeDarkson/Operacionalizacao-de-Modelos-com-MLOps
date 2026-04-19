import pandas as pd

from src.features.build_features import prepare_inference_features
from src.operations.model_package import load_model_package


def predict_from_package(package_dir: str, records) -> dict:
    model, metadata, feature_columns, _ = load_model_package(package_dir)

    if isinstance(records, dict):
        records = [records]

    df = pd.DataFrame(records)
    X = prepare_inference_features(
        df=df,
        expected_columns=feature_columns,
        id_column=metadata.get("id_column")
    )

    predictions = model.predict(X).tolist()

    response = {
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
        "reduction_method": metadata["reduction_method"],
        "predictions": predictions
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1].tolist()
        response["probabilities_class_1"] = probabilities

    return response