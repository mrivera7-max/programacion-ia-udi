# -*- coding: utf-8 -*-
"""Evaluación de modelos y selección del umbral de decisión por costo."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)


@dataclass
class EvaluationReport:
    """Métricas de un modelo sobre el conjunto de prueba."""

    model: str
    threshold: float
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    tn: int
    fp: int
    fn: int
    tp: int
    expected_cost: float

    def as_dict(self) -> dict:
        """Representación serializable (para JSON y para el dashboard)."""
        return asdict(self)


class ThresholdOptimizer:
    """Busca el umbral que minimiza el costo esperado de mantenimiento.

    En mantenimiento un falso negativo (falla no detectada) cuesta mucho más
    que un falso positivo (inspección de más). Usar 0,5 por defecto ignora esa
    asimetría; aquí el umbral se elige con datos de validación, no de prueba.

    Args:
        cost_fn: Costo de un falso negativo.
        cost_fp: Costo de un falso positivo.
    """

    def __init__(self, cost_fn: float, cost_fp: float) -> None:
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp

    def cost(self, y_true: np.ndarray, proba: np.ndarray, thr: float) -> float:
        """Costo total para un umbral dado."""
        pred = (proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        return float(fn * self.cost_fn + fp * self.cost_fp)

    def optimize(self, y_true: np.ndarray, proba: np.ndarray) -> float:
        """Barre umbrales entre 0,01 y 0,99 y devuelve el de menor costo."""
        grid = np.linspace(0.01, 0.99, 99)
        costs = [self.cost(y_true, proba, t) for t in grid]
        return float(grid[int(np.argmin(costs))])


class Evaluator:
    """Calcula el reporte de métricas de un modelo entrenado."""

    def __init__(self, optimizer: ThresholdOptimizer) -> None:
        self.optimizer = optimizer

    def evaluate(self, name: str, y_true, proba, threshold: float) -> EvaluationReport:
        """Evalúa probabilidades ``proba`` con el umbral indicado.

        Se privilegian PR-AUC y *recall* porque la clase positiva es rara
        (3,4 %): la exactitud sería de 96,6 % con un modelo que nunca alarme.
        """
        y_true = np.asarray(y_true)
        pred = (proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        return EvaluationReport(
            model=name,
            threshold=round(threshold, 2),
            roc_auc=round(roc_auc_score(y_true, proba), 4),
            pr_auc=round(average_precision_score(y_true, proba), 4),
            precision=round(precision_score(y_true, pred, zero_division=0), 4),
            recall=round(recall_score(y_true, pred), 4),
            f1=round(f1_score(y_true, pred), 4),
            tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
            expected_cost=self.optimizer.cost(y_true, proba, threshold),
        )
