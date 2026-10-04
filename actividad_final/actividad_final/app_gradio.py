# -*- coding: utf-8 -*-
"""Dashboard operativo con Gradio: simulador "¿qué pasa si?" de riesgo de falla.

El operario ajusta las condiciones de la máquina con deslizadores y recibe la
probabilidad de falla, la decisión (con el umbral de costo mínimo) y las
magnitudes físicas derivadas.

Ejecución local:   python app_gradio.py
En Colab:          !python app_gradio.py   (Gradio genera un enlace público)
"""
from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from pdm_pipeline import PredictiveMaintenancePipeline  # noqa: E402

MODEL_PATH = Path("outputs/modelo_pdm.joblib")


class FailureRiskService:
    """Servicio que encapsula el modelo entrenado y su umbral de decisión."""

    def __init__(self, path: Path = MODEL_PATH) -> None:
        if not path.exists():  # primera ejecución: entrena el pipeline
            PredictiveMaintenancePipeline().run()
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.threshold = bundle["threshold"]

    def predict(self, product_type: str, air_k: float, process_k: float,
                rpm: float, torque: float, wear: float) -> tuple[str, str]:
        """Predice el riesgo de falla para una condición de operación.

        Returns:
            Tupla (resultado en Markdown, magnitudes físicas en Markdown).
        """
        row = pd.DataFrame([{
            "type": product_type, "air_temperature_k": air_k,
            "process_temperature_k": process_k, "rotational_speed_rpm": rpm,
            "torque_nm": torque, "tool_wear_min": wear,
        }])
        proba = float(self.model.predict_proba(row)[0, 1])
        alarm = proba >= self.threshold
        verdict = ("🔴 **PROGRAMAR MANTENIMIENTO**" if alarm
                   else "🟢 **Operación normal**")
        power_kw = torque * 2 * np.pi * rpm / 60 / 1000
        physics = (f"- Potencia: **{power_kw:.2f} kW** (zona segura ≈ 3,5–9 kW)\n"
                   f"- ΔT proceso–aire: **{process_k - air_k:.1f} K**\n"
                   f"- Desgaste × torque: **{wear * torque:,.0f} N·m·min**")
        return (f"{verdict}\n\nProbabilidad de falla: **{proba:.1%}** "
                f"(umbral {self.threshold:.2f})"), physics


def build_app() -> gr.Blocks:
    """Arma la interfaz Gradio."""
    service = FailureRiskService()
    with gr.Blocks(title="Riesgo de falla – CNC") as app:
        gr.Markdown("# Simulador de riesgo de falla – fresadora CNC (AI4I 2020)")
        with gr.Row():
            with gr.Column():
                ptype = gr.Radio(["L", "M", "H"], value="M", label="Calidad del producto")
                air = gr.Slider(295, 305, value=300, step=0.1, label="Temp. aire [K]")
                proc = gr.Slider(305, 314, value=310, step=0.1, label="Temp. proceso [K]")
                rpm = gr.Slider(1150, 2900, value=1500, step=10, label="Velocidad [rpm]")
                torque = gr.Slider(3, 77, value=40, step=0.5, label="Torque [N·m]")
                wear = gr.Slider(0, 255, value=100, step=1, label="Desgaste [min]")
                btn = gr.Button("Evaluar", variant="primary")
            with gr.Column():
                out = gr.Markdown()
                phys = gr.Markdown()
        btn.click(service.predict, [ptype, air, proc, rpm, torque, wear], [out, phys])
    return app


if __name__ == "__main__":
    build_app().launch(share="google.colab" in sys.modules)
