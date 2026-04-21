"""Métricas simples para el proyecto."""

import pandas as pd


def construir_tabla_politica(agentes):
    """Unir el resumen de brazos por contexto."""
    filas = []
    for contexto, agente in agentes.items():
        resumen = agente.obtener_estadisticas().copy()
        resumen["Contexto"] = contexto
        filas.append(resumen)
    if not filas:
        return pd.DataFrame()
    return pd.concat(filas, ignore_index=True)


def evaluar_politica_bandit(agentes, tabla_agregada):
    """Comparar canal aprendido contra mejor canal histórico."""
    mejor_historico = (
        tabla_agregada.sort_values(["Contexto", "roi_promedio", "conversion_promedio"], ascending=[True, False, False])
        .drop_duplicates(subset=["Contexto"])
        .rename(columns={"Channel_Used": "mejor_canal_historico"})
    )

    filas = []
    for contexto, agente in agentes.items():
        recomendado = agente.mejor_canal_aprendido()
        fila_historica = mejor_historico[mejor_historico["Contexto"] == contexto]
        if fila_historica.empty:
            continue
        fila_historica = fila_historica.iloc[0]
        resumen = agente.obtener_estadisticas()
        fila_recomendada = resumen[resumen["canal"] == recomendado].iloc[0]
        filas.append(
            {
                "Contexto": contexto,
                "canal_recomendado_bandit": recomendado,
                "mejor_canal_historico": fila_historica["mejor_canal_historico"],
                "acierto_mejor_canal": int(recomendado == fila_historica["mejor_canal_historico"]),
                "roi_promedio_bandit": fila_recomendada["roi_promedio_observado"],
                "roi_promedio_optimo": fila_historica["roi_promedio"],
                "regret_aproximado": max(0.0, fila_historica["roi_promedio"] - fila_recomendada["roi_promedio_observado"]),
            }
        )
    return pd.DataFrame(filas)


def resumir_metricas_finales(evaluacion_df):
    """Métricas agregadas fáciles de leer."""
    if evaluacion_df.empty:
        return {}
    return {
        "n_contextos": int(evaluacion_df.shape[0]),
        "accuracy_mejor_canal": float(evaluacion_df["acierto_mejor_canal"].mean()),
        "roi_promedio_bandit": float(evaluacion_df["roi_promedio_bandit"].mean()),
        "roi_promedio_optimo": float(evaluacion_df["roi_promedio_optimo"].mean()),
        "regret_aproximado_promedio": float(evaluacion_df["regret_aproximado"].mean()),
    }
