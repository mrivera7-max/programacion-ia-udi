# -*- coding: utf-8 -*-
"""Dashboard interactivo (Plotly) con los resultados del pipeline.

Genera un único HTML autocontenido que se abre en cualquier navegador o se
muestra en Colab. Cada figura responde a una pregunta de la planta:
¿dónde falla la máquina?, ¿qué modelo usar?, ¿qué umbral?, ¿qué variable vigilar?
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROJO, AZUL, GRIS = "#C0392B", "#1F618D", "#95A5A6"


class DashboardBuilder:
    """Construye el dashboard a partir de un pipeline ya ejecutado.

    Args:
        pipeline: Instancia de ``PredictiveMaintenancePipeline`` con ``run``
            completado (necesita ``df``, ``cv_results``, ``test_frame``...).
    """

    def __init__(self, pipeline) -> None:
        self.p = pipeline
        self.df = pipeline.df

    # ---------------------------------------------------------------- figuras
    def fig_operating_map(self) -> go.Figure:
        """Mapa de operación torque vs velocidad, coloreado por falla."""
        fig = go.Figure()
        for label, color, flag in (("Sin falla", GRIS, 0), ("Falla", ROJO, 1)):
            d = self.df[self.df["machine_failure"] == flag]
            fig.add_trace(go.Scattergl(
                x=d["rotational_speed_rpm"], y=d["torque_nm"], mode="markers",
                name=label, marker=dict(color=color, size=4 + 3 * flag,
                                        opacity=0.35 + 0.6 * flag),
                hovertemplate="%{x} rpm<br>%{y:.1f} N·m<extra>" + label + "</extra>"))
        rpm = np.linspace(1150, 2900, 200)
        for watts in (3500, 9000):  # límites de potencia del mecanismo PWF
            fig.add_trace(go.Scatter(x=rpm, y=watts / (2 * np.pi * rpm / 60),
                                     mode="lines", line=dict(dash="dot", color=AZUL),
                                     name=f"P = {watts/1000:g} kW"))
        fig.update_layout(title="Mapa de operación: fallas en baja velocidad y alto torque",
                          xaxis_title="Velocidad de giro [rpm]",
                          yaxis_title="Torque [N·m]", yaxis_range=[0, 80])
        return fig

    def fig_failure_modes(self) -> go.Figure:
        """Tasa de falla por calidad de producto y conteo por modo de falla."""
        fig = make_subplots(1, 2, subplot_titles=(
            "Tasa de falla por calidad (L, M, H)", "Fallas por modo"))
        rate = self.df.groupby("type")["machine_failure"].mean().reindex(["L", "M", "H"]) * 100
        fig.add_trace(go.Bar(x=rate.index, y=rate.values, marker_color=AZUL,
                             text=[f"{v:.2f} %" for v in rate.values],
                             textposition="outside", showlegend=False), 1, 1)
        modes = {"TWF": "Desgaste herramienta", "HDF": "Disipación calor",
                 "PWF": "Potencia", "OSF": "Sobreesfuerzo", "RNF": "Aleatoria"}
        counts = self.df[[m.lower() for m in modes]].sum().values
        fig.add_trace(go.Bar(x=list(modes.values()), y=counts, marker_color=ROJO,
                             text=counts, textposition="outside",
                             showlegend=False), 1, 2)
        fig.update_yaxes(title_text="% de ciclos con falla", row=1, col=1)
        fig.update_yaxes(title_text="N.º de fallas", row=1, col=2)
        return fig

    def fig_model_comparison(self) -> go.Figure:
        """PR-AUC y recall de cada modelo en validación cruzada."""
        cv = pd.DataFrame(self.p.cv_results)
        fig = go.Figure()
        for metric, color in (("pr_auc", AZUL), ("recall", ROJO), ("precision", GRIS)):
            fig.add_trace(go.Bar(x=cv["model"], y=cv[metric], name=metric.upper(),
                                 marker_color=color, text=cv[metric].round(3),
                                 textposition="outside"))
        fig.update_layout(barmode="group", yaxis_range=[0, 1.1],
                          title="Comparación de modelos (validación cruzada 5 pliegues)")
        return fig

    def fig_confusion(self) -> go.Figure:
        """Matriz de confusión del mejor modelo en el conjunto de prueba."""
        r = self.p.test_report
        z = [[r.tn, r.fp], [r.fn, r.tp]]
        fig = go.Figure(go.Heatmap(
            z=z, x=["Predice OK", "Predice falla"], y=["Real OK", "Real falla"],
            text=z, texttemplate="%{text}", colorscale="Blues", showscale=False))
        fig.update_layout(title=f"Matriz de confusión en prueba (umbral = {r.threshold})",
                          yaxis_autorange="reversed")
        return fig

    def fig_threshold_cost(self) -> go.Figure:
        """Costo esperado en prueba según el umbral de decisión."""
        tf = self.p.test_frame
        grid = np.linspace(0.01, 0.99, 99)
        cost = [self.p.optimizer.cost(tf["y_true"].values, tf["proba"].values, t)
                for t in grid]
        fig = go.Figure(go.Scatter(x=grid, y=cost, mode="lines",
                                   line=dict(color=AZUL)))
        fig.add_vline(x=self.p.threshold, line_dash="dash", line_color=ROJO,
                      annotation_text=f"umbral elegido {self.p.threshold:.2f}")
        fig.add_vline(x=0.5, line_dash="dot", line_color=GRIS,
                      annotation_text="0,50 por defecto", annotation_position="bottom")
        fig.update_layout(title="Costo de mantenimiento vs umbral (FN = 10, FP = 1)",
                          xaxis_title="Umbral de probabilidad",
                          yaxis_title="Costo relativo en prueba")
        return fig

    def fig_importance(self) -> go.Figure:
        """Importancia por permutación (caída de PR-AUC)."""
        imp = self.p.importance.sort_values("importance")
        fig = go.Figure(go.Bar(x=imp["importance"], y=imp["feature"],
                               orientation="h", marker_color=AZUL,
                               error_x=dict(type="data", array=imp["std"])))
        fig.update_layout(title="¿Qué variable vigilar? Importancia por permutación",
                          xaxis_title="Caída de PR-AUC al permutar la variable")
        return fig

    def fig_ablation(self) -> go.Figure:
        """Aporte de las variables físicas (PR-AUC en validación cruzada)."""
        a = self.p.ablation
        labels = ["Solo variables crudas", "Crudas + físicas"]
        vals = [a.get("sin_fisicas", 0), a.get("con_fisicas", 0)]
        fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=[GRIS, AZUL],
                               text=[f"{v:.3f}" for v in vals], textposition="outside"))
        fig.update_layout(title="Ablación: ¿aportan las variables físicas? (PR-AUC CV)",
                          yaxis_range=[0, 1])
        return fig

    def fig_risk_by_wear(self) -> go.Figure:
        """Probabilidad media de falla predicha por tramo de desgaste."""
        tf = self.p.test_frame.copy()
        tf["tramo"] = pd.cut(tf["tool_wear_min"], bins=range(0, 260, 20))
        g = tf.groupby("tramo", observed=True).agg(
            proba=("proba", "mean"), real=("y_true", "mean"))
        x = [str(i) for i in g.index]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=g["proba"] * 100, name="Riesgo predicho",
                                 mode="lines+markers", line=dict(color=AZUL)))
        fig.add_trace(go.Scatter(x=x, y=g["real"] * 100, name="Tasa real",
                                 mode="lines+markers", line=dict(color=ROJO, dash="dot")))
        fig.update_layout(title="Riesgo de falla según desgaste de la herramienta",
                          xaxis_title="Desgaste [min]", yaxis_title="%")
        return fig

    # ----------------------------------------------------------------- salida
    def kpis(self) -> str:
        """Tarjetas HTML con los indicadores principales."""
        r = self.p.test_report
        base_cost = int(r.tp + r.fn) * self.p.config.cost_false_negative
        saving = 100 * (1 - r.expected_cost / base_cost)
        cards = [
            ("Tasa de falla", f"{100 * self.df['machine_failure'].mean():.2f} %"),
            ("Mejor modelo", r.model.replace("_", " ")),
            ("PR-AUC prueba", f"{r.pr_auc:.3f}"),
            ("Recall (fallas detectadas)", f"{r.recall:.1%}"),
            ("Precisión de alarmas", f"{r.precision:.1%}"),
            ("Ahorro vs. no predecir", f"{saving:.0f} %"),
        ]
        html = "".join(f"<div class='k'><span>{k}</span><b>{v}</b></div>" for k, v in cards)
        return f"<div class='kpis'>{html}</div>"

    def figures(self) -> list[go.Figure]:
        """Todas las figuras en orden de lectura."""
        figs = [self.fig_operating_map(), self.fig_failure_modes(),
                self.fig_model_comparison(), self.fig_confusion(),
                self.fig_threshold_cost(), self.fig_importance(),
                self.fig_ablation(), self.fig_risk_by_wear()]
        for f in figs:
            f.update_layout(template="plotly_white", height=430,
                            margin=dict(l=60, r=30, t=60, b=50))
        return figs

    def build(self, path: Path) -> Path:
        """Escribe el dashboard HTML autocontenido.

        Args:
            path: Ruta de salida (``.html``).

        Returns:
            La ruta escrita.
        """
        body = "".join(f.to_html(full_html=False, include_plotlyjs=(i == 0))
                       for i, f in enumerate(self.figures()))
        css = ("body{font-family:Arial,sans-serif;max-width:1100px;margin:auto;"
               "padding:16px;color:#1b2631}.kpis{display:grid;gap:10px;"
               "grid-template-columns:repeat(auto-fit,minmax(160px,1fr))}"
               ".k{border:1px solid #d5d8dc;border-radius:8px;padding:10px}"
               ".k span{display:block;font-size:12px;color:#566573}"
               ".k b{font-size:20px}")
        html = (f"<html><head><meta charset='utf-8'><title>Dashboard PdM</title>"
                f"<style>{css}</style></head><body>"
                "<h1>Mantenimiento predictivo – fresadora CNC (AI4I 2020)</h1>"
                f"{self.kpis()}{body}</body></html>")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        return path
