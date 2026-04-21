"""Implementación de Thompson Sampling siguiendo la lógica del notebook del curso.

La idea central es la misma que en `sesion5_03_bandits_personalizados.ipynb`:

- un agente por contexto
- cada brazo representa un canal
- el agente recomienda
- observa feedback
- actualiza sus parámetros Beta
"""

import numpy as np
import pandas as pd


class BanditPersonalizadoPublicidad:
    """Agente bandit para un contexto específico de publicidad."""

    def __init__(self, contexto, canales, algoritmo="thompson"):
        self.contexto = contexto
        self.canales = list(canales)
        self.n_canales = len(self.canales)
        self.algoritmo = algoritmo

        # Estadísticas observadas
        self.impresiones = np.zeros(self.n_canales)
        self.recompensas_acumuladas = np.zeros(self.n_canales)
        self.roi_acumulado = np.zeros(self.n_canales)

        # Parámetros para Thompson Sampling
        self.alphas = np.ones(self.n_canales)
        self.betas = np.ones(self.n_canales)

        self.historial = []

    def recomendar(self):
        """Recomendar un canal usando Thompson Sampling."""
        muestras = np.zeros(self.n_canales)

        for i in range(self.n_canales):
            muestras[i] = np.random.beta(self.alphas[i], self.betas[i])

        idx_canal = int(np.argmax(muestras))
        return self.canales[idx_canal], muestras

    def actualizar(self, canal, recompensa, roi_real):
        """Actualizar el agente con feedback observado."""
        idx = self.canales.index(canal)
        recompensa = float(max(0.0, min(1.0, recompensa)))

        self.impresiones[idx] += 1
        self.recompensas_acumuladas[idx] += recompensa
        self.roi_acumulado[idx] += float(roi_real)

        # Igual que en el notebook base, actualizamos alpha/beta
        if recompensa >= 0.5:
            self.alphas[idx] += 1
        else:
            self.betas[idx] += 1

        self.historial.append(
            {
                "contexto": self.contexto,
                "canal": canal,
                "recompensa": recompensa,
                "roi_real": float(roi_real),
                "paso": len(self.historial) + 1,
            }
        )

    def obtener_estadisticas(self):
        """Obtener qué aprendió el agente por brazo."""
        estadisticas = []

        for i, canal in enumerate(self.canales):
            if self.impresiones[i] > 0:
                recompensa_promedio = self.recompensas_acumuladas[i] / self.impresiones[i]
                roi_promedio = self.roi_acumulado[i] / self.impresiones[i]
            else:
                recompensa_promedio = 0.0
                roi_promedio = 0.0

            estadisticas.append(
                {
                    "canal": canal,
                    "impresiones": int(self.impresiones[i]),
                    "recompensa_promedio": recompensa_promedio,
                    "roi_promedio_observado": roi_promedio,
                    "alpha": self.alphas[i],
                    "beta": self.betas[i],
                    "muestra_esperada": self.alphas[i] / (self.alphas[i] + self.betas[i]),
                }
            )

        return pd.DataFrame(estadisticas).sort_values(
            ["muestra_esperada", "roi_promedio_observado"],
            ascending=False,
        )

    def mejor_canal_aprendido(self):
        """Obtener el canal favorito aprendido por el agente."""
        estadisticas = self.obtener_estadisticas()
        return estadisticas.iloc[0]["canal"]


def entrenar_bandits_por_contexto(observaciones, semilla=42):
    """Crear un agente por contexto y simular aprendizaje histórico."""
    np.random.seed(semilla)
    agentes = {}
    politica = []

    for contexto, canales_dict in observaciones.items():
        canales = sorted(canales_dict.keys())
        agente = BanditPersonalizadoPublicidad(contexto, canales)
        agentes[contexto] = agente

        print(f"Entrenando contexto: {contexto}")

        for canal in canales:
            for registro in canales_dict[canal]:
                canal_recomendado, _ = agente.recomendar()

                # Si el canal elegido no coincide con el registro actual, igual
                # usamos una observación disponible de ese canal para que el
                # agente aprenda en línea con la lógica del notebook del curso.
                if canal_recomendado in canales_dict and canales_dict[canal_recomendado]:
                    registro_usado = canales_dict[canal_recomendado][
                        int(agente.impresiones[agente.canales.index(canal_recomendado)]) % len(canales_dict[canal_recomendado])
                    ]
                else:
                    registro_usado = registro

                agente.actualizar(
                    canal=canal_recomendado,
                    recompensa=registro_usado["recompensa_normalizada"],
                    roi_real=registro_usado["roi"],
                )

        politica.append(
            {
                "Contexto": contexto,
                "canal_recomendado": agente.mejor_canal_aprendido(),
            }
        )

    politica_df = pd.DataFrame(politica).sort_values("Contexto").reset_index(drop=True)
    return agentes, politica_df


def entrenar_agente_global(observaciones, semilla=42):
    """Entrenar un solo agente para todos los contextos."""
    np.random.seed(semilla)

    canales = sorted(
        {
            canal
            for canales_dict in observaciones.values()
            for canal in canales_dict.keys()
        }
    )
    agente_global = BanditPersonalizadoPublicidad("global", canales)

    observaciones_globales = {canal: [] for canal in canales}
    for canales_dict in observaciones.values():
        for canal, registros in canales_dict.items():
            observaciones_globales[canal].extend(registros)

    for canal in canales:
        for registro in observaciones_globales[canal]:
            canal_recomendado, _ = agente_global.recomendar()

            if observaciones_globales.get(canal_recomendado):
                indice = int(agente_global.impresiones[agente_global.canales.index(canal_recomendado)]) % len(
                    observaciones_globales[canal_recomendado]
                )
                registro_usado = observaciones_globales[canal_recomendado][indice]
            else:
                registro_usado = registro

            agente_global.actualizar(
                canal=canal_recomendado,
                recompensa=registro_usado["recompensa_normalizada"],
                roi_real=registro_usado["roi"],
            )

    return agente_global


def comparar_bandit_contextual_vs_global(agentes_contextuales, agente_global):
    """Comparar política contextual contra agente único global."""
    filas = []
    canal_global = agente_global.mejor_canal_aprendido()
    estadisticas_globales = agente_global.obtener_estadisticas()
    fila_global = estadisticas_globales[estadisticas_globales["canal"] == canal_global].iloc[0]

    for contexto, agente in agentes_contextuales.items():
        estadisticas_contextuales = agente.obtener_estadisticas()
        canal_contextual = agente.mejor_canal_aprendido()
        fila_contextual = estadisticas_contextuales[estadisticas_contextuales["canal"] == canal_contextual].iloc[0]

        filas.append(
            {
                "Contexto": contexto,
                "canal_contextual": canal_contextual,
                "recompensa_promedio_contextual": fila_contextual["recompensa_promedio"],
                "roi_promedio_contextual": fila_contextual["roi_promedio_observado"],
                "canal_global": canal_global,
                "recompensa_promedio_global": fila_global["recompensa_promedio"],
                "roi_promedio_global": fila_global["roi_promedio_observado"],
                "coinciden": int(canal_contextual == canal_global),
                "mejor_agente": (
                    "contextual"
                    if fila_contextual["roi_promedio_observado"] >= fila_global["roi_promedio_observado"]
                    else "global"
                ),
            }
        )

    return pd.DataFrame(filas)


def recomendar_canal(agentes, tabla_agregada, target_audience, campaign_goal):
    """Recomendar canal para contexto visto o no visto."""
    contexto = f"{target_audience.strip()} | {campaign_goal.strip()}"

    if contexto in agentes:
        agente = agentes[contexto]
        return {
            "contexto": contexto,
            "estrategia": "bandit_contextual",
            "recommended_channel": agente.mejor_canal_aprendido(),
            "detalle": agente.obtener_estadisticas().to_dict(orient="records"),
        }

    solo_audiencia = tabla_agregada[tabla_agregada["Target_Audience"] == target_audience]
    if not solo_audiencia.empty:
        mejor = solo_audiencia.sort_values(["roi_promedio", "conversion_promedio"], ascending=False).iloc[0]
        return {
            "contexto": contexto,
            "estrategia": "cold_start_por_audiencia",
            "recommended_channel": mejor["Channel_Used"],
            "detalle": mejor.to_dict(),
        }

    solo_objetivo = tabla_agregada[tabla_agregada["Campaign_Goal"] == campaign_goal]
    if not solo_objetivo.empty:
        mejor = solo_objetivo.sort_values(["roi_promedio", "conversion_promedio"], ascending=False).iloc[0]
        return {
            "contexto": contexto,
            "estrategia": "cold_start_por_objetivo",
            "recommended_channel": mejor["Channel_Used"],
            "detalle": mejor.to_dict(),
        }

    mejor_global = (
        tabla_agregada.groupby("Channel_Used", as_index=False)
        .agg(roi_promedio=("roi_promedio", "mean"), conversion_promedio=("conversion_promedio", "mean"))
        .sort_values(["roi_promedio", "conversion_promedio"], ascending=False)
        .iloc[0]
    )
    return {
        "contexto": contexto,
        "estrategia": "cold_start_global",
        "recommended_channel": mejor_global["Channel_Used"],
        "detalle": mejor_global.to_dict(),
    }
