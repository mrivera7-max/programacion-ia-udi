"""Script principal del Ejercicio 2: entrenamiento monitoreado de modelos.

La lógica de entrenamiento NO contiene código de cronometraje ni de manejo de
errores: todo eso lo aporta el decorador importado del paquete ``monitoreo``.
Objetivo: predecir si un sensor está en FALLA a partir de su telemetría.
"""
from __future__ import annotations

import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from ej1_poo.cargadores import CargadorDatos
from monitoreo import historial, monitorear

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
X_COLS = ["temperatura_c", "voltaje_v", "corriente_ma"]
Datos = tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]


@monitorear(fase="datos")
def preparar_datos(ruta: str) -> Datos:
    """Carga la telemetría (reutiliza el Ejercicio 1) y separa train/test."""
    df = CargadorDatos.desde_ruta(ruta).cargar()
    y = (df["estado"] == "FALLA").astype(int)
    return train_test_split(df[X_COLS], y, test_size=0.3, random_state=42, stratify=y)


@monitorear(umbral_s=0.25, fase="entrenamiento")
def entrenar(modelo: Pipeline, x: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Ajusta el modelo recibido (función genérica: sirve para cualquier estimador)."""
    return modelo.fit(x, y)


@monitorear(fase="evaluacion")
def evaluar(modelo: Pipeline, x: pd.DataFrame, y: pd.Series) -> float:
    """Devuelve el F1 macro en el conjunto de prueba."""
    return f1_score(y, modelo.predict(x), average="macro")


@monitorear(relanzar=False, fase="entrenamiento")
def entrenar_con_error(x: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Caso de prueba: hiperparámetro inválido -> la excepción se registra y el pipeline sigue."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=-1.0)).fit(x, y)


def main() -> None:
    x_tr, x_te, y_tr, y_te = preparar_datos("datos/telemetria.csv")
    modelos = {
        "Regresión logística": make_pipeline(StandardScaler(),
                                             LogisticRegression(class_weight="balanced")),
        "Random Forest (300 árboles)": make_pipeline(
            RandomForestClassifier(n_estimators=300, random_state=42)),
    }
    for nombre, modelo in modelos.items():
        ajustado = entrenar(modelo, x_tr, y_tr)
        print(f"   -> {nombre}: F1 macro = {evaluar(ajustado, x_te, y_te):.3f}")

    resultado = entrenar_con_error(x_tr, y_tr)
    print(f"   -> Modelo con error devuelto: {resultado} (el proceso no se detuvo)")

    print("\nResumen del historial de monitoreo:")
    print(pd.DataFrame([{"función": r.funcion, "fase": r.etiquetas.get("fase"),
                         "duración_s": round(r.duracion_s, 4), "éxito": r.exito,
                         "error": (r.error or "")[:45]} for r in historial]).to_string(index=False))
    print(f"\nMetadatos preservados por functools.wraps: nombre='{entrenar.__name__}', "
          f"doc='{(entrenar.__doc__ or '').splitlines()[0]}'")


if __name__ == "__main__":
    main()
