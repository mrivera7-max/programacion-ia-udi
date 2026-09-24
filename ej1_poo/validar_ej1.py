"""Validación funcional, de eficiencia y de extensibilidad del Ejercicio 1.

1. Equivalencia: la versión POO produce exactamente lo mismo que el script
   original (línea base) para los cuatro formatos.
2. Auditoría de la sugerencia inicial de la IA: evidencia de sus defectos.
3. Eficiencia: tiempo de carga+limpieza con 200 000 filas.
4. Extensibilidad: se agrega un formato nuevo sin modificar el módulo.
"""
from __future__ import annotations

import sys
import tempfile
import timeit
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ej1_poo import procedural_original, sugerencia_inicial_ia  # noqa: E402
from ej1_poo.cargadores import CargadorDatos, ErrorCargaDatos  # noqa: E402
from generar_datos import generar_telemetria  # noqa: E402

FORMATOS = ["csv", "json", "txt", "xlsx"]


def normalizar(df: pd.DataFrame) -> pd.DataFrame:
    """Unifica tipos de columnas de texto para comparar contenido, no dtypes."""
    out = df.copy()
    for c in ("id_sensor", "timestamp", "estado"):
        out[c] = out[c].astype(str)
    return out.reset_index(drop=True)


def seccion(titulo: str) -> None:
    print(f"\n{'=' * 70}\n{titulo}\n{'=' * 70}")


def main() -> None:
    base = normalizar(procedural_original.cargar_y_limpiar("datos/telemetria.csv"))

    seccion("1) Equivalencia POO vs. script original (referencia: CSV original)")
    for fmt in FORMATOS:
        poo = normalizar(CargadorDatos.desde_ruta(f"datos/telemetria.{fmt}").cargar())
        pd.testing.assert_frame_equal(poo, base, check_dtype=False)
        print(f"  .{fmt:<5} OK  {poo.shape}  (idéntico a la línea base)")

    seccion("2) Auditoría de la sugerencia inicial de la IA")
    print(f"  {'formato':<8}{'filas':>7}{'nulos':>7}  tipo timestamp   ¿igual a base?")
    for fmt in FORMATOS:
        ia = sugerencia_inicial_ia.obtener_cargador(f"datos/telemetria.{fmt}").cargar()
        igual = len(ia) == len(base) and int(ia.isna().sum().sum()) == 0
        print(f"  .{fmt:<7}{len(ia):>7}{int(ia.isna().sum().sum()):>7}  "
              f"{str(ia['timestamp'].dtype):<16} {'sí' if igual else 'NO'}")
    vacio = sugerencia_inicial_ia.obtener_cargador("no_existe.csv").cargar()
    print(f"  Archivo inexistente -> la IA devuelve DataFrame vacío de forma "
          f"silenciosa: shape={vacio.shape}")
    try:
        CargadorDatos.desde_ruta("no_existe.csv").cargar()
    except ErrorCargaDatos as exc:
        print(f"  Versión corregida -> lanza ErrorCargaDatos: {exc}")

    seccion("3) Eficiencia con 200 000 filas (mejor de 3 ejecuciones)")
    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(tmp) / "grande.csv"
        generar_telemetria(200_000).to_csv(ruta, index=False)
        t_proc = min(timeit.repeat(lambda: procedural_original.cargar_y_limpiar(str(ruta)),
                                   number=1, repeat=3))
        t_poo = min(timeit.repeat(lambda: CargadorDatos.desde_ruta(ruta).cargar(),
                                  number=1, repeat=3))
    print(f"  Estructurado: {t_proc:.3f} s | POO: {t_poo:.3f} s | "
          f"relación POO/estructurado = {t_poo / t_proc:.2f}")

    seccion("4) Extensibilidad: nuevo formato .psv (separado por '|') sin editar el módulo")

    @CargadorDatos.registrar(".psv")
    class CargadorPSV(CargadorDatos):
        def _leer(self) -> pd.DataFrame:
            return pd.read_csv(self.ruta, sep="|", dtype=str)

    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(tmp) / "telemetria.psv"
        pd.read_csv("datos/telemetria.csv", dtype=str).to_csv(ruta, sep="|", index=False)
        df = CargadorDatos.desde_ruta(ruta).cargar()
        print(f"  {type(CargadorDatos.desde_ruta(ruta)).__name__} registrado -> {df.shape} OK")


if __name__ == "__main__":
    main()
