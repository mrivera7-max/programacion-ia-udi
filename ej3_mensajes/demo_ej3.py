"""Demostración del sistema de análisis de mensajes (Factory + Strategy)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ej3_mensajes.estrategias import AnalizadorMensajes, EstrategiaAnalisis, Resultado  # noqa: E402
from ej3_mensajes.fabrica import FabricaEstrategias  # noqa: E402

MUESTRA = [
    "Falla crítica en la fuente de alimentación, huele a quemado",
    "El proyector del aula 101 no enciende",
    "Gracias por el apoyo en la práctica de ayer",
]


def main() -> None:
    # 1) La estrategia se elige por configuración externa (Factory).
    config = json.loads('{"estrategia": "prioridad_reglas"}')
    analizador = AnalizadorMensajes(FabricaEstrategias.desde_config(config))
    print("Estrategias registradas:", FabricaEstrategias.disponibles())

    # 2) Se cambia el algoritmo en ejecución sin tocar el analizador (Strategy).
    etiquetado = pd.read_csv("datos/mensajes_etiquetados.csv")
    estrategias = [
        analizador.estrategia,
        FabricaEstrategias.crear("sentimiento"),
        FabricaEstrategias.crear("prioridad_ml", textos=etiquetado["mensaje"],
                                 etiquetas=etiquetado["prioridad_real"]),
    ]
    filas = []
    for est in estrategias:
        analizador.estrategia = est
        for msg, r in zip(MUESTRA, analizador.analizar_lote(MUESTRA), strict=True):
            filas.append({"estrategia": r.estrategia, "mensaje": msg[:40],
                          "etiqueta": r.etiqueta, "confianza": round(r.confianza, 2)})
    print(pd.DataFrame(filas).to_string(index=False))

    # 3) Extensión sin modificar código existente (principio abierto/cerrado).
    class EstrategiaLongitud(EstrategiaAnalisis):
        nombre = "longitud"

        def analizar(self, mensaje: str) -> Resultado:
            n = len(mensaje.split())
            return Resultado(self.nombre, "extenso" if n > 8 else "breve", 1.0, f"{n} palabras")

    FabricaEstrategias.registrar("longitud", EstrategiaLongitud)
    analizador.estrategia = FabricaEstrategias.crear("longitud")
    print("\nNueva estrategia 'longitud':",
          [(r.etiqueta, r.detalle) for r in analizador.analizar_lote(MUESTRA)])

    try:
        FabricaEstrategias.crear("inexistente")
    except KeyError as exc:
        print("Validación de la fábrica:", exc)


if __name__ == "__main__":
    main()
