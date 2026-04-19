import numpy as np
import pandas as pd


def clean_numeric_columns(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        if col != id_column:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace("%", "", regex=False)
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
                .replace(["nan", "None", ""], np.nan)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def create_target(
    df: pd.DataFrame,
    target_source_column: str,
    target_column: str
) -> pd.DataFrame:
    df = df.dropna(subset=[target_source_column]).copy()

    threshold = df[target_source_column].median()
    df[target_column] = (df[target_source_column] > threshold).astype(int)

    return df

def build_feature_matrix(
    df: pd.DataFrame,
    id_column: str,
    target_source_column: str,
    target_column: str
):
    drop_cols = [id_column, target_source_column, target_column]

    ids = df[id_column].copy()
    X = df.drop(columns=drop_cols).copy()
    y = df[target_column].copy()

    X = X.select_dtypes(include=[np.number]).copy()

    return X, y, ids

def prepare_dataset(
    df: pd.DataFrame,
    id_column: str,
    target_source_column: str,
    target_column: str
):
    df_clean = clean_numeric_columns(df, id_column=id_column)
    df_target = create_target(
        df_clean,
        target_source_column=target_source_column,
        target_column=target_column
    )
    X, y, ids = build_feature_matrix(
        df_target,
        id_column=id_column,
        target_source_column=target_source_column,
        target_column=target_column
    )

    return X, y, ids, df_target

def prepare_inference_features(
    df: pd.DataFrame,
    expected_columns: list,
    id_column: str | None = None
) -> pd.DataFrame:

    df = df.copy()

    if id_column and id_column in df.columns:
        df = df.drop(columns=[id_column])

    for col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace("%", "", regex=False)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
            .replace(["nan", "None", ""], np.nan)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in expected_columns:
        if col not in df.columns:
            df[col] = np.nan

    df = df[expected_columns].copy()
    return df