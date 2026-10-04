# -*- coding: utf-8 -*-
"""Capa de datos: carga, normalización de nombres y validación de esquema."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from .config import UCI_URL

logger = logging.getLogger(__name__)


class DataLoader:
    """Carga el dataset AI4I 2020 desde disco o, si no existe, desde UCI.

    Args:
        path: Ruta local del archivo CSV.
        url: URL de respaldo para descargar el archivo.
    """

    def __init__(self, path: Path, url: str = UCI_URL) -> None:
        self.path = Path(path)
        self.url = url

    def load(self) -> pd.DataFrame:
        """Lee el CSV y devuelve un DataFrame con columnas normalizadas.

        Returns:
            DataFrame con nombres de columna en ``snake_case`` y unidades
            como sufijo (p. ej. ``torque_nm``).
        """
        source = self.path if self.path.exists() else self.url
        logger.info("Leyendo datos desde %s", source)
        # encoding 'utf-8-sig' elimina el BOM que trae el archivo original.
        df = pd.read_csv(source, encoding="utf-8-sig")
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.path, index=False)
        df.columns = [self.normalize_column(c) for c in df.columns]
        logger.info("Datos cargados: %d filas x %d columnas", *df.shape)
        return df

    @staticmethod
    def normalize_column(name: str) -> str:
        """Convierte un encabezado a ``snake_case`` usando expresiones regulares.

        Ejemplo: ``'Rotational speed [rpm]'`` -> ``'rotational_speed_rpm'``.

        Args:
            name: Encabezado original.

        Returns:
            Encabezado normalizado.
        """
        name = re.sub(r"\[(.*?)\]", r"_\1", name)       # '[rpm]' -> '_rpm'
        name = re.sub(r"[^0-9a-zA-Z]+", "_", name)      # símbolos -> '_'
        return re.sub(r"_+", "_", name).strip("_").lower()


class DataValidator:
    """Verifica el esquema y la calidad mínima de los datos antes de modelar.

    La validación temprana evita que un CSV corrupto produzca un modelo
    silenciosamente equivocado (principio *fail fast*).
    """

    REQUIRED = (
        "type", "air_temperature_k", "process_temperature_k",
        "rotational_speed_rpm", "torque_nm", "tool_wear_min", "machine_failure",
    )
    #: Patrón de los identificadores de producto: L, M o H + 5 dígitos.
    PRODUCT_ID = re.compile(r"^[LMH]\d{5}$")

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida columnas, nulos, dominio de valores y formato de IDs.

        Args:
            df: Datos normalizados por :class:`DataLoader`.

        Returns:
            El mismo DataFrame si supera todas las verificaciones.

        Raises:
            ValueError: Si falta una columna o hay valores fuera de dominio.
        """
        missing = set(self.REQUIRED) - set(df.columns)
        if missing:
            raise ValueError(f"Columnas faltantes: {sorted(missing)}")
        nulls = int(df[list(self.REQUIRED)].isna().sum().sum())
        if nulls:
            raise ValueError(f"Se encontraron {nulls} valores nulos")
        if not df["type"].isin(["L", "M", "H"]).all():
            raise ValueError("La columna 'type' solo admite L, M o H")
        bad_ids = (~df["product_id"].astype(str).str.match(self.PRODUCT_ID)).sum()
        if bad_ids:
            logger.warning("%d IDs de producto con formato inesperado", bad_ids)
        if (df[["rotational_speed_rpm", "torque_nm", "tool_wear_min"]] < 0).any().any():
            raise ValueError("Hay magnitudes físicas negativas")
        logger.info("Validación superada; tasa de falla = %.2f %%",
                    100 * df["machine_failure"].mean())
        return df
