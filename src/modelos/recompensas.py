"""Funciones de recompensa simples para el bandit."""


def recompensa_roi_normalizada(roi, roi_min, roi_max):
    """Normalizar ROI al rango [0, 1]."""
    rango = max(roi_max - roi_min, 1e-9)
    return max(0.0, min(1.0, (float(roi) - roi_min) / rango))


def recompensa_binaria_por_umbral(roi, umbral):
    """Recompensa binaria basada en umbral de ROI."""
    return 1.0 if float(roi) >= float(umbral) else 0.0


def recompensa_compuesta(roi, acquisition_cost, roi_min, roi_max, costo_min, costo_max):
    """Mezclar ROI alto con penalización por costo alto."""
    rango_roi = max(roi_max - roi_min, 1e-9)
    rango_costo = max(costo_max - costo_min, 1e-9)

    roi_norm = max(0.0, min(1.0, (float(roi) - roi_min) / rango_roi))
    costo_norm = max(0.0, min(1.0, (float(acquisition_cost) - costo_min) / rango_costo))
    return max(0.0, min(1.0, 0.8 * roi_norm + 0.2 * (1.0 - costo_norm)))
