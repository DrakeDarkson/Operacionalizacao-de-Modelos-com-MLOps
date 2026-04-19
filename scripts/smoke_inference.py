import yaml

from src.data.load_data import load_data
from src.features.build_features import prepare_dataset
from src.operations.inference import predict_from_package
from src.operations.model_package import find_latest_package


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    config = load_config()

    package_dir = find_latest_package(
        package_root=config["deployment"]["package_dir"],
        package_name=config["deployment"]["package_name"],
        model_name=config["deployment"]["model_name"],
        reduction_name=config["deployment"]["reduction_method"]
    )

    df = load_data(config["data"]["raw_path"])
    X, _, _, _ = prepare_dataset(
        df=df,
        id_column=config["data"]["id_column"],
        target_source_column=config["data"]["target_source_column"],
        target_column=config["data"]["target_column"]
    )

    sample_records = X.head(3).to_dict(orient="records")
    result = predict_from_package(package_dir, sample_records)

    print("Pacote carregado de:", package_dir)
    print("Resultado da inferência:")
    print(result)

    assert "predictions" in result
    assert len(result["predictions"]) == 3

    print("Smoke test de inferência executado com sucesso.")


if __name__ == "__main__":
    main()