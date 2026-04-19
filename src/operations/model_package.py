import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def resolve_version(version: str | None) -> str:
    if version and version.lower() != "auto":
        return version
    return datetime.now().strftime("v%Y%m%d_%H%M%S")


def build_package_dir(
    package_root: str,
    package_name: str,
    model_name: str,
    reduction_name: str,
    version: str
) -> Path:
    return Path(package_root) / package_name / model_name / reduction_name / version


def make_json_serializable(obj):

    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [make_json_serializable(value) for value in obj]

    if isinstance(obj, tuple):
        return [make_json_serializable(value) for value in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    return obj


def save_model_package(
    trained_model,
    feature_columns: list,
    metadata: dict,
    reference_features: pd.DataFrame,
    package_root: str,
    package_name: str,
    model_name: str,
    reduction_name: str,
    version: str
) -> str:
    version = resolve_version(version)
    package_dir = build_package_dir(
        package_root=package_root,
        package_name=package_name,
        model_name=model_name,
        reduction_name=reduction_name,
        version=version
    )
    package_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(trained_model, package_dir / "model.joblib")

    serializable_metadata = make_json_serializable(metadata)
    with open(package_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(serializable_metadata, file, ensure_ascii=False, indent=2)

    serializable_feature_columns = make_json_serializable(feature_columns)
    with open(package_dir / "feature_columns.json", "w", encoding="utf-8") as file:
        json.dump(serializable_feature_columns, file, ensure_ascii=False, indent=2)

    reference_features.to_csv(package_dir / "reference_features.csv", index=False)

    return str(package_dir)


def load_model_package(package_dir: str):
    package_path = Path(package_dir)

    model = joblib.load(package_path / "model.joblib")

    with open(package_path / "metadata.json", "r", encoding="utf-8") as file:
        metadata = json.load(file)

    with open(package_path / "feature_columns.json", "r", encoding="utf-8") as file:
        feature_columns = json.load(file)

    reference_features = pd.read_csv(package_path / "reference_features.csv")

    return model, metadata, feature_columns, reference_features


def find_latest_package(
    package_root: str,
    package_name: str,
    model_name: str,
    reduction_name: str
) -> str:
    base_dir = Path(package_root) / package_name / model_name / reduction_name
    if not base_dir.exists():
        raise FileNotFoundError(f"Diretório de pacotes não encontrado: {base_dir}")

    versions = [path for path in base_dir.iterdir() if path.is_dir()]
    if not versions:
        raise FileNotFoundError(f"Nenhuma versão encontrada em: {base_dir}")

    latest = sorted(versions)[-1]
    return str(latest)