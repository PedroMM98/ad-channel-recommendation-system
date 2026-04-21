"""Adaptación simple de enfoque comparativo.

El dataset no tiene una matriz usuario-item clásica. Por eso no se implementa
filtrado colaborativo tradicional. En su lugar se deja una utilidad que explica
esa limitación y sugiere usar agregados por contexto como aproximación.
"""


def analizar_viabilidad_filtrado_colaborativo():
    """Explicar por qué CF clásico no es ideal en este dataset."""
    return {
        "aplica_directamente": False,
        "motivo": (
            "El dataset no contiene interacciones usuario-item repetidas ni "
            "una matriz explícita de usuarios contra canales o anuncios."
        ),
        "adaptacion_sugerida": (
            "Usar agregación por contexto (audiencia + objetivo) y comparar "
            "canales como brazos del bandit."
        ),
    }
