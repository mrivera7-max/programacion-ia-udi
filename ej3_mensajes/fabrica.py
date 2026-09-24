"""Patrón Factory (con registro): crea estrategias a partir de un nombre o de
una configuración externa, desacoplando al cliente de las clases concretas.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ej3_mensajes.estrategias import (
    EstrategiaAnalisis,
    EstrategiaModeloML,
    EstrategiaPrioridadReglas,
    EstrategiaSentimientoLexico,
)

Constructor = Callable[..., EstrategiaAnalisis]


class FabricaEstrategias:
    """Registro nombre -> constructor. Nuevas estrategias no modifican la fábrica."""

    _registro: dict[str, Constructor] = {}

    @classmethod
    def registrar(cls, nombre: str, constructor: Constructor) -> None:
        if nombre in cls._registro:
            raise ValueError(f"La estrategia '{nombre}' ya está registrada")
        cls._registro[nombre] = constructor

    @classmethod
    def crear(cls, nombre: str, **parametros: Any) -> EstrategiaAnalisis:
        """Instancia la estrategia pedida.

        Raises:
            KeyError: Si el nombre no está registrado (mensaje con las opciones).
        """
        try:
            return cls._registro[nombre](**parametros)
        except KeyError:
            raise KeyError(f"Estrategia '{nombre}' desconocida. "
                           f"Disponibles: {sorted(cls._registro)}") from None

    @classmethod
    def desde_config(cls, config: dict[str, Any]) -> EstrategiaAnalisis:
        """Crea la estrategia desde un dict (p. ej., leído de JSON/YAML)."""
        return cls.crear(config["estrategia"], **config.get("parametros", {}))

    @classmethod
    def disponibles(cls) -> list[str]:
        return sorted(cls._registro)


FabricaEstrategias.registrar("prioridad_reglas", EstrategiaPrioridadReglas)
FabricaEstrategias.registrar("sentimiento", EstrategiaSentimientoLexico)
FabricaEstrategias.registrar("prioridad_ml", EstrategiaModeloML)
