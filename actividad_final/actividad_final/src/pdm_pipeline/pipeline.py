# -*- coding: utf-8 -*-
"""Orquestador (patrón Facade) del pipeline de mantenimiento predictivo.

Flujo: cargar -> validar -> dividir -> comparar modelos con validación cruzada
-> elegir umbral por costo -> evaluar en prueba -> explicar -> persistir ->
generar dashboard.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from .config import PipelineConfig
from .data import DataLoader, DataValidator
from .evaluation import Evaluator, ThresholdOptimizer
from .models import ModelFactory

logger = logging.getLogger(__name__)

#: Columnas que se excluyen para evitar fuga de información (*data leakage*):
#: los modos de falla TWF..RNF son consecuencia de la falla, no causa.
LEAKAGE = ["udi", "product_id", "twf", "hdf", "pwf", "osf", "rnf"]


class PredictiveMaintenancePipeline:
    """Fachada que coordina todas las etapas del pipeline.

    Args:
        config: Parámetros de la corrida.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.optimizer = ThresholdOptimizer(self.config.cost_false_negative,
                                            self.config.cost_false_positive)
        self.evaluator = Evaluator(self.optimizer)
        self.df: pd.DataFrame | None = None
        self.cv_results: list[dict] = []
        self.best_name: str | None = None
        self.best_model = None
        self.threshold: float = 0.5
        self.test_report = None
        self.importance: pd.DataFrame | None = None
        self.test_frame: pd.DataFrame | None = None
        self.ablation: dict[str, float] = {}

    # ------------------------------------------------------------------ datos
    def load_data(self) -> pd.DataFrame:
        """Carga y valida el dataset."""
        df = DataLoader(self.config.data_path).load()
        self.df = DataValidator().validate(df)
        return self.df

    def split(self):
        """Divide en entrenamiento y prueba de forma estratificada."""
        X = self.df.drop(columns=[self.config.target] + LEAKAGE)
        y = self.df[self.config.target]
        return train_test_split(X, y, test_size=self.config.test_size,
                                stratify=y, random_state=self.config.random_state)

    # --------------------------------------------------------------- modelado
    def compare_models(self, X_train, y_train) -> pd.DataFrame:
        """Compara las estrategias con validación cruzada estratificada.

        Las probabilidades fuera de pliegue (*out-of-fold*) sirven para dos
        cosas: estimar PR-AUC sin tocar el conjunto de prueba y elegir el
        umbral de costo mínimo de cada modelo.
        """
        cv = StratifiedKFold(self.config.cv_folds, shuffle=True,
                             random_state=self.config.random_state)
        for name in self.config.models:
            estimator = ModelFactory.create(name, self.config.random_state).build()
            oof = cross_val_predict(estimator, X_train, y_train, cv=cv,
                                    method="predict_proba", n_jobs=-1)[:, 1]
            thr = self.optimizer.optimize(y_train.values, oof)
            report = self.evaluator.evaluate(name, y_train, oof, thr)
            self.cv_results.append(report.as_dict())
            logger.info("CV %-20s PR-AUC=%.3f recall=%.3f umbral=%.2f",
                        name, report.pr_auc, report.recall, thr)
        table = pd.DataFrame(self.cv_results).sort_values("pr_auc", ascending=False)
        self.best_name = table.iloc[0]["model"]
        self.threshold = float(table.iloc[0]["threshold"])
        return table

    def ablation_study(self, X_train, y_train) -> dict[str, float]:
        """Mide el aporte de las variables físicas (con vs. sin) en el mejor modelo.

        Responde a la pregunta metodológica: ¿la ingeniería de características
        guiada por el dominio mejora el desempeño o es decorativa?
        """
        cv = StratifiedKFold(self.config.cv_folds, shuffle=True,
                             random_state=self.config.random_state)
        for label, physics in (("con_fisicas", True), ("sin_fisicas", False)):
            est = ModelFactory.create(self.best_name, self.config.random_state,
                                      use_physics=physics).build()
            oof = cross_val_predict(est, X_train, y_train, cv=cv,
                                    method="predict_proba", n_jobs=-1)[:, 1]
            self.ablation[label] = round(average_precision_score(y_train, oof), 4)
        logger.info("Ablación PR-AUC: %s", self.ablation)
        return self.ablation

    def fit_best(self, X_train, y_train):
        """Reentrena el mejor modelo con todo el conjunto de entrenamiento."""
        self.best_model = ModelFactory.create(self.best_name,
                                              self.config.random_state).build()
        self.best_model.fit(X_train, y_train)
        return self.best_model

    def evaluate_test(self, X_test, y_test):
        """Evalúa una sola vez en prueba con el umbral fijado en validación."""
        proba = self.best_model.predict_proba(X_test)[:, 1]
        self.test_report = self.evaluator.evaluate(self.best_name, y_test,
                                                   proba, self.threshold)
        self.test_frame = X_test.assign(y_true=y_test.values, proba=proba,
                                        y_pred=(proba >= self.threshold).astype(int))
        logger.info("Prueba: %s", self.test_report.as_dict())
        return self.test_report

    def explain(self, X_test, y_test) -> pd.DataFrame:
        """Importancia por permutación medida como caída de PR-AUC."""
        result = permutation_importance(
            self.best_model, X_test, y_test, n_repeats=10,
            scoring="average_precision", random_state=self.config.random_state,
            n_jobs=-1,
        )
        self.importance = (pd.DataFrame({"feature": X_test.columns,
                                         "importance": result.importances_mean,
                                         "std": result.importances_std})
                           .sort_values("importance", ascending=False))
        return self.importance

    # ------------------------------------------------------------ persistencia
    def save(self) -> Path:
        """Guarda modelo, umbral y métricas en ``output_dir``."""
        out = self.config.output_dir
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.best_model, "threshold": self.threshold},
                    out / "modelo_pdm.joblib")
        payload = {"cv": self.cv_results, "test": self.test_report.as_dict(),
                   "ablation": self.ablation,
                   "importance": self.importance.to_dict(orient="records")}
        (out / "metricas.json").write_text(json.dumps(payload, indent=2,
                                                      ensure_ascii=False))
        self.test_frame.to_csv(out / "predicciones_prueba.csv", index=False)
        return out

    # ------------------------------------------------------------------ main
    def run(self) -> dict:
        """Ejecuta el pipeline completo y devuelve un resumen."""
        self.load_data()
        X_train, X_test, y_train, y_test = self.split()
        table = self.compare_models(X_train, y_train)
        self.ablation_study(X_train, y_train)
        self.fit_best(X_train, y_train)
        self.evaluate_test(X_test, y_test)
        self.explain(X_test, y_test)
        self.save()
        from .dashboard import DashboardBuilder  # import tardío: plotly es opcional
        DashboardBuilder(self).build(self.config.output_dir / "dashboard.html")
        from .static_figures import StaticFigureExporter
        StaticFigureExporter(self, self.config.output_dir / "figuras").export_all()
        return {"cv": table, "test": self.test_report.as_dict(),
                "ablation": self.ablation,
                "importance": self.importance}


def main() -> None:
    """Punto de entrada por línea de comandos: ``python -m pdm_pipeline``."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    summary = PredictiveMaintenancePipeline().run()
    print(summary["cv"].to_string(index=False))
    print(json.dumps(summary["test"], indent=2))


if __name__ == "__main__":
    main()
