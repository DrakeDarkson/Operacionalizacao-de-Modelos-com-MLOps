import os

from flask import Flask, jsonify, request
import yaml

from src.operations.inference import predict_from_package
from src.operations.model_package import find_latest_package


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


config = load_config()

MODEL_PACKAGE_DIR = os.getenv("MODEL_PACKAGE_DIR")
if not MODEL_PACKAGE_DIR:
    MODEL_PACKAGE_DIR = find_latest_package(
        package_root=config["deployment"]["package_dir"],
        package_name=config["deployment"]["package_name"],
        model_name=config["deployment"]["model_name"],
        reduction_name=config["deployment"]["reduction_method"]
    )

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_package_dir": MODEL_PACKAGE_DIR
    })


@app.post("/predict")
def predict():
    payload = request.get_json()

    if payload is None:
        return jsonify({"error": "JSON inválido"}), 400

    records = payload.get("records", payload)
    result = predict_from_package(MODEL_PACKAGE_DIR, records)
    return jsonify(result)


if __name__ == "__main__":
    app.run(
        host=config["service"]["host"],
        port=config["service"]["port"],
        debug=False
    )