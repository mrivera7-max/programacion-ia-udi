"""Carga y limpieza de telemetría con Programación Orientada a Objetos.

Versión corregida tras auditar la sugerencia inicial de la IA. Diseño:

* ``CargadorDatos`` (ABC) implementa el patrón *Template Method*: ``cargar()``
  fija el flujo invariable (leer -> validar esquema -> limpiar) y delega en las
  subclases solo el paso que varía: ``_leer()``.
* Cada subclase maneja un formato (herencia) y todas se usan a través de la
  misma interfaz (polimorfismo).
* Un registro por decorador (``@registrar``) reemplaza la cadena if/elif: un
  formato nuevo se agrega creando una clase, sin tocar código existente (OCP).
* La limpieza vive en ``LimpiadorTelemetria`` (SRP): se puede probar y
  reutilizar sin leer archivos.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pandas as pd

logger = logging.getLogger(__name__)

COLUMNAS_ESPERADAS: tuple[str, ...] = (
    "id_sensor", "timestamp", "temperatura_c", "voltaje_v", "corriente_ma", "estado",
)
COLUMNAS_NUMERICAS: tuple[str, ...] = ("temperatura_c", "voltaje_v", "corriente_ma")


class ErrorCargaDatos(Exception):
    """Error de dominio: el archivo no puede cargarse o no cumple el esquema."""


@dataclass(frozen=True)
class LimpiadorTelemetria:
    """Reglas de limpieza de la telemetría (independientes del formato).

    Attributes:
        temp_min: Límite inferior válido de temperatura (°C, exclusivo).
        temp_max: Límite superior válido de temperatura (°C, exclusivo).
    """

    temp_min: float = -40.0
    temp_max: float = 125.0

    def limpiar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica la limpieza en el orden correcto: tipar primero, filtrar después.

        Args:
            df: Datos crudos con las columnas de ``COLUMNAS_ESPERADAS``.

        Returns:
            DataFrame limpio, sin duplicados ni nulos y con tipos consistentes.
        """
        df = df.drop_duplicates().copy()
        for col in COLUMNAS_NUMERICAS:
            # Coma decimal -> punto; cualquier texto no numérico ("N/A") -> NaN.
            texto = df[col].astype(str).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(texto, errors="coerce")
        df["estado"] = df["estado"].astype(str).str.strip().str.upper()
        df = df[df["temperatura_c"].between(self.temp_min, self.temp_max,
                                            inclusive="neither")]
        # dropna DESPUÉS de convertir tipos: así se eliminan también los "N/A".
        return df.dropna().reset_index(drop=True)


class CargadorDatos(ABC):
    """Interfaz común y flujo de carga (Template Method).

    Args:
        ruta: Archivo a cargar.
        limpiador: Estrategia de limpieza inyectada (por defecto la estándar).
    """

    _registro: ClassVar[dict[str, type[CargadorDatos]]] = {}

    def __init__(self, ruta: str | Path, limpiador: LimpiadorTelemetria | None = None) -> None:
        self.ruta = Path(ruta)
        self.limpiador = limpiador or LimpiadorTelemetria()

    # ------------------------------------------------------------------ registro
    @classmethod
    def registrar(cls, *extensiones: str):
        """Decorador de clase que asocia extensiones con una subclase."""
        def _decorar(subclase: type[CargadorDatos]) -> type[CargadorDatos]:
            for ext in extensiones:
                cls._registro[ext.lower()] = subclase
            return subclase
        return _decorar

    @classmethod
    def desde_ruta(cls, ruta: str | Path, **kwargs: object) -> CargadorDatos:
        """Fábrica: devuelve el cargador adecuado según la extensión del archivo.

        Raises:
            ErrorCargaDatos: Si la extensión no tiene cargador registrado.
        """
        ext = Path(ruta).suffix.lower()
        try:
            return cls._registro[ext](ruta, **kwargs)  # type: ignore[arg-type]
        except KeyError:
            soportados = ", ".join(sorted(cls._registro))
            raise ErrorCargaDatos(f"Formato '{ext}' no soportado. Use: {soportados}") from None

    # ------------------------------------------------------------ flujo común
    def cargar(self) -> pd.DataFrame:
        """Lee, valida y limpia el archivo. Los errores se propagan, no se ocultan.

        Raises:
            ErrorCargaDatos: Si el archivo no existe, no se puede leer o le
                faltan columnas.
        """
        if not self.ruta.exists():
            raise ErrorCargaDatos(f"No existe el archivo: {self.ruta}")
        try:
            crudo = self._leer()
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            raise ErrorCargaDatos(f"No se pudo leer {self.ruta.name}: {exc}") from exc
        self._validar_esquema(crudo)
        limpio = self.limpiador.limpiar(crudo)
        logger.info("%s: %d filas crudas -> %d filas limpias",
                    self.ruta.name, len(crudo), len(limpio))
        return limpio

    @staticmethod
    def _validar_esquema(df: pd.DataFrame) -> None:
        faltantes = set(COLUMNAS_ESPERADAS) - set(df.columns)
        if faltantes:
            raise ErrorCargaDatos(f"Faltan columnas: {sorted(faltantes)}")

    @abstractmethod
    def _leer(self) -> pd.DataFrame:
        """Lee el archivo y devuelve TODAS las columnas como texto (dtype=str).

        Leer como texto garantiza que todas las subclases entreguen el mismo
        esquema; la conversión de tipos es responsabilidad del limpiador.
        """


# ------------------------------------------------------------------ subclases
@CargadorDatos.registrar(".csv")
class CargadorCSV(CargadorDatos):
    """Archivos separados por comas."""

    def _leer(self) -> pd.DataFrame:
        return pd.read_csv(self.ruta, dtype=str, encoding="utf-8")


@CargadorDatos.registrar(".txt", ".tsv")
class CargadorTXT(CargadorDatos):
    """Archivos de texto tabulados."""

    def _leer(self) -> pd.DataFrame:
        return pd.read_csv(self.ruta, sep="\t", dtype=str, encoding="utf-8")


@CargadorDatos.registrar(".json")
class CargadorJSON(CargadorDatos):
    """Archivos JSON orientados a registros (lista de objetos)."""

    def _leer(self) -> pd.DataFrame:
        with self.ruta.open(encoding="utf-8") as f:
            registros = json.load(f)
        # dtype=str evita la inferencia de tipos de read_json (timestamp -> datetime).
        return pd.DataFrame.from_records(registros).astype(str).replace({"nan": None, "None": None})


@CargadorDatos.registrar(".xlsx", ".xls")
class CargadorExcel(CargadorDatos):
    """Libros de Excel (primera hoja)."""

    def _leer(self) -> pd.DataFrame:
        return pd.read_excel(self.ruta, dtype=str)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    for archivo in sorted(Path("datos").glob("telemetria.*")):
        datos = CargadorDatos.desde_ruta(archivo).cargar()  # polimorfismo
        print(f"{archivo.name:18s} {datos.shape}  nulos={int(datos.isna().sum().sum())}")
