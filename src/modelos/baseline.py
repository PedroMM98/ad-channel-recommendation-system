"""Baselines simples para comparar el bandit."""

import numpy as np

from src.data.features import agregar_metricas_por_contexto_y_canal


def baseline_mejor_canal_global(df):
    """Elegir el canal con mayor ROI promedio global."""
    tabla = (
        df.groupby("Channel_Used", as_index=False)
        .agg(roi_promedio=("ROI", "mean"), conversion_promedio=("Conversion_Rate", "mean"))
        .sort_values(["roi_promedio", "conversion_promedio"], ascending=False)
    )
    return tabla.iloc[0].to_dict()


def baseline_mejor_canal_por_contexto(df):
    """Elegir el mejor canal histórico por contexto."""
    resumen = agregar_metricas_por_contexto_y_canal(df)
    politica = resumen.sort_values(
        ["Contexto", "roi_promedio", "conversion_promedio"],
        ascending=[True, False, False],
    ).drop_duplicates(subset=["Contexto"])
    return politica[["Contexto", "Channel_Used", "roi_promedio", "conversion_promedio", "costo_promedio"]]


def baseline_aleatorio(canales, semilla=42):
    """Seleccionar un canal aleatorio reproducible."""
    generador = np.random.default_rng(semilla)
    return generador.choice(list(canales))
