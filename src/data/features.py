"""Features y agregaciones del proyecto."""

import pandas as pd


def crear_contexto(df):
    """Crear columna de contexto audiencia + objetivo."""
    datos = df.copy()
    datos["Contexto"] = datos["Target_Audience"].str.strip() + " | " + datos["Campaign_Goal"].str.strip()
    return datos


def agregar_metricas_por_contexto_y_canal(df):
    """Crear tabla agregada para análisis y baselines."""
    resumen = (
        df.groupby(["Contexto", "Target_Audience", "Campaign_Goal", "Channel_Used"], as_index=False)
        .agg(
            n_observaciones=("Channel_Used", "size"),
            roi_promedio=("ROI", "mean"),
            roi_mediano=("ROI", "median"),
            conversion_promedio=("Conversion_Rate", "mean"),
            costo_promedio=("Acquisition_Cost", "mean"),
            clicks_promedio=("Clicks", "mean"),
            impresiones_promedio=("Impressions", "mean"),
            engagement_promedio=("Engagement_Score", "mean"),
        )
    )
    return resumen.sort_values(["Contexto", "roi_promedio"], ascending=[True, False]).reset_index(drop=True)


def crear_tabla_contexto(df):
    """Tabla del mejor canal observado por contexto."""
    resumen = agregar_metricas_por_contexto_y_canal(df)
    mejor = resumen.sort_values(
        ["Contexto", "roi_promedio", "conversion_promedio"],
        ascending=[True, False, False],
    ).drop_duplicates(subset=["Contexto"])
    return mejor.reset_index(drop=True)


def preparar_observaciones_bandit(df):
    """Convertir dataframe en observaciones por contexto y canal."""
    observaciones = {}
    datos = df.sort_values("Date").copy()

    roi_min = float(datos["ROI"].min())
    roi_max = float(datos["ROI"].max())
    rango = max(roi_max - roi_min, 1e-9)

    for fila in datos.itertuples(index=False):
        contexto = fila.Contexto
        canal = fila.Channel_Used
        recompensa_normalizada = (float(fila.ROI) - roi_min) / rango
        registro = {
            "roi": float(fila.ROI),
            "conversion_rate": float(fila.Conversion_Rate),
            "acquisition_cost": float(fila.Acquisition_Cost),
            "recompensa_normalizada": float(recompensa_normalizada),
        }
        observaciones.setdefault(contexto, {})
        observaciones[contexto].setdefault(canal, [])
        observaciones[contexto][canal].append(registro)

    return observaciones
