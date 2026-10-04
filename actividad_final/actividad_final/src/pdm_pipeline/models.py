# -*- coding: utf-8 -*-
"""Modelos predictivos organizados con los patrones Strategy y Factory.

* **Strategy**: cada algoritmo es una estrategia intercambiable que expone la
  misma interfaz (``build``). El pipeline no depende de un algoritmo concreto.
* **Factory**: ``ModelFactory`` crea la estrategia a partir de un nombre, de
  modo que agregar un modelo nuevo no obliga a tocar el orquestador
  (principio abierto/cerrado).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import PhysicsFeatureEngineer


class ModelStrategy(ABC):
    """Interfaz común de todas las estrategias de modelado."""

    name: str = "base"

    def __init__(self, random_state: int = 42, use_physics: bool = True) -> None:
        self.random_state = random_state
        self.use_physics = use_physics  # False -> solo variables crudas (ablación)

    @abstractmethod
    def classifier(self):
        """Devuelve el estimador de scikit-learn sin entrenar."""

    def build(self) -> Pipeline:
        """Arma el pipeline completo: variables físicas, preprocesamiento y modelo.

        Returns:
            ``sklearn.pipeline.Pipeline`` listo para ``fit``/``predict_proba``.
        """
        numeric = (PhysicsFeatureEngineer.numeric_columns() if self.use_physics
                   else list(PhysicsFeatureEngineer.NUMERIC_IN))
        preprocess = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric),
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["type"]),
            ]
        )
        return Pipeline(
            steps=[
                ("physics", PhysicsFeatureEngineer()),
                ("preprocess", preprocess),
                ("model", self.classifier()),
            ]
        )


class LogisticRegressionStrategy(ModelStrategy):
    """Línea base interpretable; ``class_weight`` compensa el desbalance."""

    name = "logistic_regression"

    def classifier(self):
        return LogisticRegression(max_iter=2000, class_weight="balanced",
                                  random_state=self.random_state)


class RandomForestStrategy(ModelStrategy):
    """Ensamble de árboles; captura no linealidades e interacciones."""

    name = "random_forest"

    def classifier(self):
        return RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                      class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=self.random_state)


class GradientBoostingStrategy(ModelStrategy):
    """Boosting por histogramas; suele ser el más preciso en datos tabulares."""

    name = "gradient_boosting"

    def classifier(self):
        return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400,
                                              class_weight="balanced",
                                              random_state=self.random_state)


class ModelFactory:
    """Fábrica que traduce un nombre en una estrategia concreta."""

    _registry: dict[str, type[ModelStrategy]] = {
        cls.name: cls for cls in (LogisticRegressionStrategy,
                                  RandomForestStrategy,
                                  GradientBoostingStrategy)
    }

    @classmethod
    def create(cls, name: str, random_state: int = 42,
               use_physics: bool = True) -> ModelStrategy:
        """Instancia la estrategia registrada con ``name``.

        Raises:
            KeyError: Si el nombre no está registrado.
        """
        if name not in cls._registry:
            raise KeyError(f"Modelo '{name}' no registrado: {list(cls._registry)}")
        return cls._registry[name](random_state=random_state, use_physics=use_physics)

    @classmethod
    def register(cls, strategy: type[ModelStrategy]) -> type[ModelStrategy]:
        """Decorador para registrar nuevas estrategias sin modificar la fábrica."""
        cls._registry[strategy.name] = strategy
        return strategy
