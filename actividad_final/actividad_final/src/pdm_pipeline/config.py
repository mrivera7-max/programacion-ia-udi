# -*- coding: utf-8 -*-
"""Configuración centralizada del pipeline de mantenimiento predictivo.

Concentrar los parámetros en una dataclass inmutable evita "números mágicos"
dispersos en el código y hace reproducible cada corrida (misma semilla,
mismas rutas, mismos costos).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: URL oficial del dataset AI4I 2020 (UCI Machine Learning Repository).
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"


@dataclass(frozen=True)
class PipelineConfig:
    """Parámetros de ejecución del pipeline.

    Attributes:
        data_path: Ruta local del CSV. Si no existe se descarga de ``UCI_URL``.
        output_dir: Carpeta donde se guardan modelo, métricas y dashboard.
        target: Nombre (normalizado) de la variable objetivo.
        test_size: Proporción del conjunto de prueba (hold-out).
        random_state: Semilla para reproducibilidad.
        cv_folds: Número de pliegues de la validación cruzada estratificada.
        cost_false_negative: Costo relativo de NO detectar una falla
            (parada no programada, daño de la pieza y la herramienta).
        cost_false_positive: Costo relativo de una inspección innecesaria.
        models: Estrategias de modelado a comparar (claves del ``ModelFactory``).
    """

    data_path: Path = Path("data/ai4i2020.csv")
    output_dir: Path = Path("outputs")
    target: str = "machine_failure"
    test_size: float = 0.20
    random_state: int = 42
    cv_folds: int = 5
    cost_false_negative: float = 10.0
    cost_false_positive: float = 1.0
    models: tuple[str, ...] = field(
        default=("logistic_regression", "random_forest", "gradient_boosting")
    )
