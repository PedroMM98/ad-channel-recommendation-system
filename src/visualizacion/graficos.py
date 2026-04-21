"""Gráficos reutilizables del proyecto."""

import matplotlib.pyplot as plt
import seaborn as sns


def grafico_roi_por_canal(df):
    """Boxplot simple de ROI por canal."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="Channel_Used", y="ROI", ax=ax)
    ax.set_title("Distribución de ROI por canal")
    ax.set_xlabel("Canal")
    ax.set_ylabel("ROI")
    return fig, ax


def grafico_heatmap_roi(tabla_agregada):
    """Heatmap de ROI promedio por audiencia y objetivo."""
    tabla = (
        tabla_agregada.groupby(["Target_Audience", "Campaign_Goal"], as_index=False)
        .agg(roi_promedio=("roi_promedio", "max"))
        .pivot(index="Target_Audience", columns="Campaign_Goal", values="roi_promedio")
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(tabla, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    ax.set_title("Mejor ROI promedio por audiencia y objetivo")
    return fig, ax
