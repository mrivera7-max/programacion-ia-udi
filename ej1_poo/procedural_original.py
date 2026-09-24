"""Versión ORIGINAL (paradigma estructurado) del proceso de carga y limpieza.

Así operaba el script antes de la refactorización: una sola función con
ramas if/elif por formato y la limpieza "pegada" al final. Se conserva sin
cambios de diseño como línea base (baseline) para comparar funcionalidad y
rendimiento con la versión orientada a objetos.
"""
import json

import pandas as pd


def cargar_y_limpiar(ruta):
    # 1) Lectura: una rama por formato. Agregar un formato = editar esta función.
    if ruta.endswith(".csv"):
        df = pd.read_csv(ruta)
    elif ruta.endswith(".json"):
        with open(ruta, encoding="utf-8") as f:
            df = pd.DataFrame(json.load(f))
    elif ruta.endswith(".txt"):
        df = pd.read_csv(ruta, sep="\t")
    elif ruta.endswith(".xlsx"):
        df = pd.read_excel(ruta)
    else:
        print("Formato no soportado")
        return None

    # 2) Limpieza: reglas mezcladas con la lectura (sin separación de responsabilidades).
    df = df.drop_duplicates()
    for col in ["temperatura_c", "voltaje_v", "corriente_ma"]:
        df[col] = df[col].astype(str).str.replace(",", ".")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["estado"] = df["estado"].str.strip().str.upper()
    df = df[(df["temperatura_c"] > -40) & (df["temperatura_c"] < 125)]
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    for r in ["datos/telemetria.csv", "datos/telemetria.json",
              "datos/telemetria.txt", "datos/telemetria.xlsx"]:
        d = cargar_y_limpiar(r)
        print(r, d.shape)
