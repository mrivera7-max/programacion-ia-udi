# -*- coding: utf-8 -*-
"""Ingeniería de características con fundamento físico.

Las variables derivadas responden a los mecanismos de falla documentados
para el dataset (Matzka, 2020):

* Disipación de calor (HDF): depende de la diferencia entre la temperatura de
  proceso y la del aire, y de la velocidad de giro.
* Potencia (PWF): P = T * omega, con omega = 2*pi*n/60 [rad/s].
* Sobreesfuerzo (OSF): producto del desgaste de la herramienta por el torque.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PhysicsFeatureEngineer(BaseEstimator, TransformerMixin):
    """Transformador compatible con scikit-learn que agrega variables físicas.

    Al heredar de ``TransformerMixin`` se integra dentro de un ``Pipeline``,
    por lo que la misma transformación se aplica en entrenamiento, validación
    cruzada y predicción en producción (evita *training-serving skew*).
    """

    NUMERIC_IN = (
        "air_temperature_k", "process_temperature_k",
        "rotational_speed_rpm", "torque_nm", "tool_wear_min",
    )
    DERIVED = ("temp_diff_k", "power_w", "strain_nm_min")

    def fit(self, X: pd.DataFrame, y=None) -> "PhysicsFeatureEngineer":
        """No aprende parámetros; existe por compatibilidad con la API."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Devuelve una copia de ``X`` con las variables derivadas.

        Args:
            X: DataFrame con las columnas de ``NUMERIC_IN`` y ``type``.

        Returns:
            DataFrame ampliado (no modifica el original).
        """
        X = X.copy()
        X["temp_diff_k"] = X["process_temperature_k"] - X["air_temperature_k"]
        omega = 2 * np.pi * X["rotational_speed_rpm"] / 60.0
        X["power_w"] = X["torque_nm"] * omega
        X["strain_nm_min"] = X["tool_wear_min"] * X["torque_nm"]
        return X

    @classmethod
    def numeric_columns(cls) -> list[str]:
        """Lista de columnas numéricas que salen del transformador."""
        return list(cls.NUMERIC_IN) + list(cls.DERIVED)
