"""Limpieza simple y tipado de columnas."""

import pandas as pd


def _limpiar_moneda(valor):
    if pd.isna(valor):
        return None
    return float(str(valor).replace("$", "").replace(",", "").strip())


def limpiar_dataset_publicidad(df):
    """Estandarizar tipos y nombres mínimos necesarios."""
    datos = df.copy()
    datos["Acquisition_Cost"] = datos["Acquisition_Cost"].apply(_limpiar_moneda)
    datos["Conversion_Rate"] = pd.to_numeric(datos["Conversion_Rate"], errors="coerce")
    datos["ROI"] = pd.to_numeric(datos["ROI"], errors="coerce")
    datos["Clicks"] = pd.to_numeric(datos["Clicks"], errors="coerce")
    datos["Impressions"] = pd.to_numeric(datos["Impressions"], errors="coerce")
    datos["Engagement_Score"] = pd.to_numeric(datos["Engagement_Score"], errors="coerce")
    datos["Date"] = pd.to_datetime(datos["Date"], errors="coerce")
    datos = datos.dropna(
        subset=["Target_Audience", "Campaign_Goal", "Channel_Used", "ROI", "Acquisition_Cost"]
    ).reset_index(drop=True)
    return datos
