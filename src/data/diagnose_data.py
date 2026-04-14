import pandas as pd


def diagnose_dataset(df: pd.DataFrame) -> dict:

    report = {}

    report["shape"] = df.shape

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    report["missing_values"] = missing.to_dict()

    report["dtypes"] = df.dtypes.astype(str).to_dict()

    return report


def print_diagnosis(report: dict):
    print("=== DIAGNÓSTICO DO DATASET ===")
    print(f"Shape: {report['shape']}\n")

    print("Valores ausentes:")
    if report["missing_values"]:
        for col, val in report["missing_values"].items():
            print(f"{col}: {val}")
    else:
        print("Nenhum valor ausente")

    print("\nTipos de dados:")
    for col, dtype in report["dtypes"].items():
        print(f"{col}: {dtype}")