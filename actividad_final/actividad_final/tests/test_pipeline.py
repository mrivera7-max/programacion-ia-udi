# -*- coding: utf-8 -*-
"""Pruebas unitarias mínimas (ejecutar con ``pytest -q`` desde la raíz)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdm_pipeline.data import DataLoader, DataValidator  # noqa: E402
from pdm_pipeline.evaluation import ThresholdOptimizer  # noqa: E402
from pdm_pipeline.features import PhysicsFeatureEngineer  # noqa: E402
from pdm_pipeline.models import ModelFactory  # noqa: E402


def test_normalize_column():
    """El regex convierte encabezados con unidades a snake_case."""
    assert DataLoader.normalize_column("Rotational speed [rpm]") == "rotational_speed_rpm"
    assert DataLoader.normalize_column("Torque [Nm]") == "torque_nm"


def test_power_feature():
    """P = T * 2*pi*n/60: 40 N·m a 1500 rpm = 6283 W."""
    row = pd.DataFrame([{"air_temperature_k": 300, "process_temperature_k": 310,
                         "rotational_speed_rpm": 1500, "torque_nm": 40,
                         "tool_wear_min": 10, "type": "M"}])
    out = PhysicsFeatureEngineer().transform(row)
    assert out["power_w"].iloc[0] == pytest.approx(6283.19, rel=1e-4)
    assert out["temp_diff_k"].iloc[0] == 10


def test_validator_rejects_missing_column():
    """Un esquema incompleto debe fallar temprano."""
    with pytest.raises(ValueError):
        DataValidator().validate(pd.DataFrame({"type": ["L"]}))


def test_threshold_prefers_recall_when_fn_expensive():
    """Con FN caro, el umbral óptimo debe quedar por debajo de 0,5."""
    y = np.array([0] * 90 + [1] * 10)
    proba = np.r_[np.linspace(0, 0.4, 90), np.linspace(0.3, 0.9, 10)]
    assert ThresholdOptimizer(cost_fn=10, cost_fp=1).optimize(y, proba) < 0.5


def test_factory_unknown_model():
    """La fábrica informa nombres no registrados."""
    with pytest.raises(KeyError):
        ModelFactory.create("svm_cuantico")
