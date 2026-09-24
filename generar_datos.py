"""Genera los conjuntos de datos sintéticos usados en el taller.

Escenario: telemetría de sensores IoT de un laboratorio de electrónica
(temperatura, voltaje y corriente) con "suciedad" deliberada para ejercitar
la limpieza: duplicados, nulos, textos basura, comas decimales, valores
fuera de rango y categorías inconsistentes.

Uso:
    python generar_datos.py            # dataset pequeño (demo)
    python generar_datos.py --filas N  # dataset de N filas (pruebas de carga)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DIR_DATOS = Path(__file__).parent / "datos"
SEMILLA = 42


def generar_telemetria(n_filas: int, semilla: int = SEMILLA) -> pd.DataFrame:
    """Crea un DataFrame de telemetría con errores típicos de captura."""
    rng = np.random.default_rng(semilla)
    df = pd.DataFrame(
        {
            "id_sensor": rng.choice(["S01", "S02", "S03", "S04"], n_filas),
            "timestamp": pd.date_range("2026-08-01", periods=n_filas, freq="min").astype(str),
            "temperatura_c": rng.normal(35, 5, n_filas).round(2),
            "voltaje_v": rng.normal(5.0, 0.15, n_filas).round(3),
            "corriente_ma": rng.normal(120, 20, n_filas).round(1),
        }
    )
    # Estado dependiente de la física del sensor: sobretemperatura, desviación
    # de la tensión nominal o sobrecorriente elevan la probabilidad de FALLA.
    z = ((df["temperatura_c"] - 40) / 3 + (df["voltaje_v"] - 5.0).abs() / 0.1
         + (df["corriente_ma"] - 140) / 10 - 1.5)
    falla = rng.random(n_filas) < 1 / (1 + np.exp(-z))
    grafias_ok = rng.choice(["OK", "ok ", " Ok"], n_filas)       # mayúsculas/espacios
    grafias_falla = rng.choice(["FALLA", "falla", " Falla"], n_filas)
    df["estado"] = np.where(falla, grafias_falla, grafias_ok)
    df = df.astype({"temperatura_c": object, "voltaje_v": object})

    # --- Inyección controlada de suciedad (≈2 % de filas por tipo) ---
    k = max(1, n_filas // 50)
    idx = rng.choice(n_filas, size=4 * k, replace=False)
    df.loc[idx[:k], "temperatura_c"] = np.nan                  # nulos
    df.loc[idx[k:2 * k], "voltaje_v"] = "N/A"                  # texto basura
    df.loc[idx[2 * k:3 * k], "temperatura_c"] = 999.0          # fuera de rango
    df.loc[idx[3 * k:], "voltaje_v"] = (                       # coma decimal
        df.loc[idx[3 * k:], "voltaje_v"].astype(str).str.replace(".", ",", regex=False)
    )
    duplicados = df.sample(n=k, random_state=semilla)          # duplicados
    return pd.concat([df, duplicados], ignore_index=True)


MENSAJES = {
    "turno_manana.txt": [
        ("URGENTE: el osciloscopio del laboratorio 3 genera chispas al encender", "alta"),
        ("Falla crítica en la fuente de alimentación, huele a quemado", "alta"),
        ("El servidor de simulación está caído y no responde", "alta"),
        ("Por favor revisar el multímetro 12, marca lecturas inestables", "media"),
        ("Se requiere actualizar MATLAB en las estaciones del aula 204", "media"),
        ("Gracias por el apoyo en la práctica de ayer", "baja"),
        ("Recordatorio: reunión de docentes el viernes", "baja"),
        ("El generador de funciones presenta un error intermitente", "media"),
        ("Cortocircuito en la protoboard de la mesa 5, riesgo eléctrico", "alta"),
        ("Solicito préstamo de cables banana para el martes", "baja"),
    ],
    "turno_tarde.txt": [
        ("Incendio pequeño controlado en el laboratorio, revisar extintores de inmediato", "alta"),
        ("La impresora 3D tiene un retraso en la cola de impresión", "media"),
        ("Consulta: ¿hay kits de Arduino disponibles?", "baja"),
        ("Pérdida de datos en el servidor de prácticas, backup falló", "alta"),
        ("El proyector del aula 101 no enciende", "media"),
        ("Felicitaciones al semillero por el póster", "baja"),
        ("Revisar la conexión de red del laboratorio de comunicaciones, está lenta", "media"),
        ("Descarga eléctrica leve al tocar el chasis del equipo 7", "alta"),
        ("Informativo: nuevo horario de monitorías publicado", "baja"),
        ("Advertencia: batería del UPS al 10 %", "alta"),
    ],
    "turno_noche.txt": [
        ("Error al compilar el firmware del microcontrolador", "media"),
        ("Pendiente revisar el inventario de resistencias", "baja"),
        ("Sensor de humo activado en la bodega de equipos", "alta"),
        ("El software de licencias está lento pero funciona", "media"),
        ("Saludo cordial y agradecimiento por la capacitación", "baja"),
        ("Acceso no autorizado detectado en el servidor del laboratorio", "alta"),
        ("Solicitud de mantenimiento preventivo para las fuentes DC", "media"),
        ("Duda sobre el formato del informe de laboratorio", "baja"),
        ("Cable de alimentación pelado en la mesa 2, peligro", "alta"),
        ("Actualización del sistema operativo programada para el sábado", "baja"),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filas", type=int, default=500)
    parser.add_argument("--destino", type=Path, default=DIR_DATOS)
    args = parser.parse_args()
    args.destino.mkdir(parents=True, exist_ok=True)

    df = generar_telemetria(args.filas)
    df.to_csv(args.destino / "telemetria.csv", index=False)
    df.to_json(args.destino / "telemetria.json", orient="records", indent=1, force_ascii=False)
    df.to_csv(args.destino / "telemetria.txt", sep="\t", index=False)
    df.to_excel(args.destino / "telemetria.xlsx", index=False)

    # Mensajes para los ejercicios 3 y 4 (un mensaje por línea)
    dir_msg = args.destino / "mensajes"
    dir_msg.mkdir(exist_ok=True)
    etiquetas: list[dict[str, str]] = []
    for archivo, filas in MENSAJES.items():
        (dir_msg / archivo).write_text("\n".join(m for m, _ in filas) + "\n", encoding="utf-8")
        etiquetas += [{"archivo": archivo, "mensaje": m, "prioridad_real": p} for m, p in filas]
    pd.DataFrame(etiquetas).to_csv(args.destino / "mensajes_etiquetados.csv", index=False)
    print(f"Datos generados en {args.destino.resolve()} ({len(df)} filas de telemetría)")


if __name__ == "__main__":
    main()
