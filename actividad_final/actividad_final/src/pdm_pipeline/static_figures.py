# -*- coding: utf-8 -*-
"""Versión estática (PNG, matplotlib) del dashboard para el informe en PDF.

El dashboard Plotly es interactivo; el informe necesita imágenes fijas. Esta
clase reutiliza los mismos datos del pipeline para que ambas vistas coincidan.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROJO, AZUL, GRIS = "#C0392B", "#1F618D", "#95A5A6"


class StaticFigureExporter:
    """Exporta las figuras clave del dashboard como PNG.

    Args:
        pipeline: Pipeline ya ejecutado.
        out_dir: Carpeta de destino.
    """

    def __init__(self, pipeline, out_dir: Path) -> None:
        self.p = pipeline
        self.df = pipeline.df
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                             "axes.spines.right": False})

    def _save(self, fig, name: str) -> Path:
        path = self.out / f"{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return path

    def operating_map(self) -> Path:
        """Torque vs velocidad con curvas de potencia constante."""
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ok = self.df[self.df.machine_failure == 0]
        ko = self.df[self.df.machine_failure == 1]
        ax.scatter(ok.rotational_speed_rpm, ok.torque_nm, s=3, c=GRIS, alpha=.3,
                   label="Sin falla")
        ax.scatter(ko.rotational_speed_rpm, ko.torque_nm, s=10, c=ROJO,
                   label="Falla")
        rpm = np.linspace(1150, 2900, 200)
        for w in (3500, 9000):
            ax.plot(rpm, w / (2 * np.pi * rpm / 60), "--", c=AZUL, lw=1)
            ax.annotate(f"{w / 1000:g} kW", (2750, w / (2 * np.pi * 2750 / 60)),
                        color=AZUL, va="bottom")
        ax.set(xlabel="Velocidad de giro [rpm]", ylabel="Torque [N·m]", ylim=(0, 80),
               title="Mapa de operación: fallas en baja velocidad y alto torque")
        ax.legend(loc="upper right")
        return self._save(fig, "fig1_mapa_operacion")

    def failure_modes(self) -> Path:
        """Tasa por calidad y conteo por modo de falla."""
        fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6))
        rate = self.df.groupby("type").machine_failure.mean().reindex(["L", "M", "H"]) * 100
        a.bar(rate.index, rate.values, color=AZUL)
        for i, v in enumerate(rate.values):
            a.text(i, v + .05, f"{v:.2f} %", ha="center")
        a.set(title="Tasa de falla por calidad", ylabel="% de ciclos")
        modes = {"hdf": "Calor", "osf": "Sobreesf.", "pwf": "Potencia",
                 "twf": "Desgaste", "rnf": "Aleatoria"}
        c = self.df[list(modes)].sum()
        b.bar(list(modes.values()), c.values, color=ROJO)
        for i, v in enumerate(c.values):
            b.text(i, v + 1, str(v), ha="center")
        b.set(title="Fallas por modo (pueden coexistir)", ylabel="N.º de fallas")
        return self._save(fig, "fig2_modos_falla")

    def model_comparison(self) -> Path:
        """PR-AUC, recall y precisión por modelo en CV."""
        cv = pd.DataFrame(self.p.cv_results).sort_values("pr_auc")
        fig, ax = plt.subplots(figsize=(8, 3.6))
        y = np.arange(len(cv))
        for k, (m, col) in enumerate((("pr_auc", AZUL), ("recall", ROJO),
                                      ("precision", GRIS))):
            ax.barh(y + (k - 1) * .27, cv[m], .27, color=col, label=m.upper())
            for yi, v in zip(y, cv[m]):
                ax.text(v + .01, yi + (k - 1) * .27, f"{v:.3f}", va="center", fontsize=8)
        ax.set_yticks(y, [s.replace("_", " ") for s in cv.model])
        ax.set(xlim=(0, 1.1), title="Validación cruzada estratificada (5 pliegues)")
        ax.legend(loc="lower right")
        return self._save(fig, "fig3_comparacion_modelos")

    def confusion_and_cost(self) -> Path:
        """Matriz de confusión en prueba y costo vs umbral."""
        r, tf = self.p.test_report, self.p.test_frame
        fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 3.8),
                                   gridspec_kw={"width_ratios": [1, 1.6]})
        z = np.array([[r.tn, r.fp], [r.fn, r.tp]])
        a.imshow(z, cmap="Blues", norm=matplotlib.colors.LogNorm())
        for i in range(2):
            for j in range(2):
                a.text(j, i, z[i, j], ha="center", va="center", fontsize=13,
                       color="white" if z[i, j] > 100 else "black")
        a.set_xticks([0, 1], ["Predice OK", "Predice falla"])
        a.set_yticks([0, 1], ["Real OK", "Real falla"])
        a.set_title(f"Prueba (n = 2000, umbral {r.threshold})")
        grid = np.linspace(.01, .99, 99)
        cost = [self.p.optimizer.cost(tf.y_true.values, tf.proba.values, t) for t in grid]
        b.plot(grid, cost, c=AZUL)
        b.axvline(self.p.threshold, c=ROJO, ls="--")
        b.axvline(.5, c=GRIS, ls=":")
        b.text(self.p.threshold + .01, max(cost) * .9, f"elegido {self.p.threshold:.2f}",
               color=ROJO)
        b.text(.51, max(cost) * .75, "0,50 por defecto", color="#555")
        b.set(xlabel="Umbral de probabilidad", ylabel="Costo relativo",
              title="Costo en prueba (FN = 10, FP = 1)")
        return self._save(fig, "fig4_confusion_costo")

    def importance_and_ablation(self) -> Path:
        """Importancia por permutación y ablación de variables físicas."""
        imp = self.p.importance.sort_values("importance")
        a_ = self.p.ablation
        fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 3.6),
                                   gridspec_kw={"width_ratios": [1.6, 1]})
        a.barh(imp.feature, imp.importance, xerr=imp["std"], color=AZUL)
        a.set(xlabel="Caída de PR-AUC al permutar", title="Importancia por permutación")
        vals = [a_["sin_fisicas"], a_["con_fisicas"]]
        b.bar(["Crudas", "Crudas +\nfísicas"], vals, color=[GRIS, AZUL])
        for i, v in enumerate(vals):
            b.text(i, v + .02, f"{v:.3f}", ha="center")
        b.set(ylim=(0, 1), title="Ablación (PR-AUC CV)")
        return self._save(fig, "fig5_importancia_ablacion")

    def risk_by_wear(self) -> Path:
        """Riesgo predicho y tasa real por tramo de desgaste."""
        tf = self.p.test_frame.copy()
        tf["tramo"] = pd.cut(tf.tool_wear_min, bins=range(0, 260, 20))
        g = tf.groupby("tramo", observed=True).agg(p=("proba", "mean"), r=("y_true", "mean"))
        x = [f"{int(i.left)}–{int(i.right)}" for i in g.index]
        fig, ax = plt.subplots(figsize=(8, 3.4))
        ax.plot(x, g.p * 100, "o-", c=AZUL, label="Riesgo predicho")
        ax.plot(x, g.r * 100, "s:", c=ROJO, label="Tasa real")
        ax.set(xlabel="Desgaste de herramienta [min]", ylabel="%",
               title="El riesgo crece después de 200 min de desgaste")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        return self._save(fig, "fig6_riesgo_desgaste")

    def export_all(self) -> list[Path]:
        """Exporta todas las figuras y devuelve sus rutas."""
        return [self.operating_map(), self.failure_modes(), self.model_comparison(),
                self.confusion_and_cost(), self.importance_and_ablation(),
                self.risk_by_wear()]
