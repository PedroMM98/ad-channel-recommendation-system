"""Carga de datos del proyecto."""

from pathlib import Path

import pandas as pd

from src.config import DATASET_PATH


def cargar_dataset(ruta_csv=None):
    """Cargar el dataset principal desde CSV."""
    ruta = Path(ruta_csv) if ruta_csv else DATASET_PATH
    return pd.read_csv(ruta)


def resumir_dataset(df):
    """Generar un resumen simple del dataset."""
    return {
        "n_filas": int(df.shape[0]),
        "n_columnas": int(df.shape[1]),
        "columnas": list(df.columns),
        "nulos_por_columna": df.isna().sum().to_dict(),
    }
