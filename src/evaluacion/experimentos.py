"""Flujo principal para ejecutar el sistema de punta a punta."""

from src.data.carga_datos import cargar_dataset
from src.data.features import agregar_metricas_por_contexto_y_canal, crear_contexto, preparar_observaciones_bandit
from src.data.limpieza import limpiar_dataset_publicidad
from src.evaluacion.metricas import construir_tabla_politica, evaluar_politica_bandit, resumir_metricas_finales
from src.modelos.bandit_thompson import (
    comparar_bandit_contextual_vs_global,
    entrenar_agente_global,
    entrenar_bandits_por_contexto,
    recomendar_canal,
)
from src.modelos.baseline import baseline_mejor_canal_global, baseline_mejor_canal_por_contexto
from src.modelos.filtro_colaborativo import analizar_viabilidad_filtrado_colaborativo


def ejecutar_flujo_completo(ruta_csv):
    """Ejecutar carga, preparación, baseline, bandit y evaluación."""
    df = cargar_dataset(ruta_csv)
    df = limpiar_dataset_publicidad(df)
    df = crear_contexto(df)

    tabla_agregada = agregar_metricas_por_contexto_y_canal(df)
    observaciones = preparar_observaciones_bandit(df)
    agentes, politica_aprendida = entrenar_bandits_por_contexto(observaciones)
    agente_global = entrenar_agente_global(observaciones)
    evaluacion = evaluar_politica_bandit(agentes, tabla_agregada)
    metricas = resumir_metricas_finales(evaluacion)
    comparacion_global = comparar_bandit_contextual_vs_global(agentes, agente_global)

    audiencia_ejemplo = df["Target_Audience"].iloc[0]
    objetivo_ejemplo = df["Campaign_Goal"].iloc[0]

    return {
        "df_limpio": df,
        "tabla_agregada": tabla_agregada,
        "baseline_global": baseline_mejor_canal_global(df),
        "baseline_contextual": baseline_mejor_canal_por_contexto(df),
        "agentes": agentes,
        "agente_global": agente_global,
        "politica_aprendida": politica_aprendida,
        "tabla_politica_brazos": construir_tabla_politica(agentes),
        "evaluacion": evaluacion,
        "metricas_finales": metricas,
        "comparacion_contextual_vs_global": comparacion_global,
        "recomendacion_ejemplo": recomendar_canal(agentes, tabla_agregada, audiencia_ejemplo, objetivo_ejemplo),
        "cold_start_audiencia_nueva": recomendar_canal(agentes, tabla_agregada, "Teens 13-17", objetivo_ejemplo),
        "cold_start_objetivo_nuevo": recomendar_canal(agentes, tabla_agregada, audiencia_ejemplo, "Lead Generation"),
        "cold_start_total": recomendar_canal(agentes, tabla_agregada, "Teens 13-17", "Lead Generation"),
        "analisis_cf": analizar_viabilidad_filtrado_colaborativo(),
    }
