"""Decorador ``@monitorear``: tiempo de ejecución y gestión de excepciones.

Aprovecha que en Python las funciones son objetos de primera clase: se reciben
como argumento, se envuelven en otra función (clausura) y se devuelven. La
lógica de monitoreo queda escrita UNA sola vez (DRY) y separada de la lógica
de entrenamiento (SRP).
"""
from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, overload

_logger = logging.getLogger("monitoreo")


@dataclass(frozen=True)
class RegistroEjecucion:
    """Métrica inmutable de una ejecución monitoreada."""

    funcion: str
    inicio: datetime
    duracion_s: float
    exito: bool
    error: str | None = None
    etiquetas: dict[str, str] = field(default_factory=dict)


# Historial en memoria; en producción se reemplazaría por un exportador
# (MLflow, Prometheus, base de datos) sin tocar el código de entrenamiento.
historial: list[RegistroEjecucion] = []


# Sobrecargas: informan al verificador de tipos (mypy) que la firma de la
# función decorada se conserva (ParamSpec P) y que el retorno solo puede ser
# None cuando relanzar=False.
@overload
def monitorear[**P, R](func: Callable[P, R], /) -> Callable[P, R]: ...
@overload
def monitorear[**P, R](*, relanzar: Literal[True] = ..., umbral_s: float | None = ...,
                       logger: logging.Logger | None = ...,
                       **etiquetas: str) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
@overload
def monitorear[**P, R](*, relanzar: Literal[False], umbral_s: float | None = ...,
                       logger: logging.Logger | None = ...,
                       **etiquetas: str) -> Callable[[Callable[P, R]], Callable[P, R | None]]: ...


def monitorear(func: Callable[..., Any] | None = None, /, *, relanzar: bool = True,
               umbral_s: float | None = None, logger: logging.Logger | None = None,
               **etiquetas: str) -> Any:
    """Registra duración y excepciones de la función decorada.

    Se puede usar con o sin argumentos: ``@monitorear`` o
    ``@monitorear(relanzar=False, umbral_s=2.0, fase="entrenamiento")``.

    Args:
        func: Función a decorar (solo en la forma sin paréntesis).
        relanzar: Si es ``True`` la excepción se registra y se propaga
            (comportamiento seguro por defecto); si es ``False`` se registra y
            la función devuelve ``None`` para que el pipeline continúe.
        umbral_s: Si la ejecución supera este tiempo se emite una advertencia.
        logger: Logger a usar; por defecto ``logging.getLogger("monitoreo")``.
        **etiquetas: Metadatos libres guardados con cada registro.

    Returns:
        La función envuelta, con el mismo nombre, docstring y firma.
    """
    log = logger or _logger

    def decorador(f: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(f)  # preserva __name__, __doc__, __wrapped__
        def envoltura(*args: Any, **kwargs: Any) -> Any:
            inicio = datetime.now()
            t0 = time.perf_counter()          # reloj monotónico de alta resolución
            try:
                resultado = f(*args, **kwargs)
            except Exception as exc:          # nunca BaseException (Ctrl+C, SystemExit)
                dur = time.perf_counter() - t0
                historial.append(RegistroEjecucion(f.__qualname__, inicio, dur, False,
                                                   f"{type(exc).__name__}: {exc}", etiquetas))
                log.error("✗ %s falló tras %.3f s -> %s: %s",
                          f.__qualname__, dur, type(exc).__name__, exc,
                          exc_info=log.isEnabledFor(logging.DEBUG))
                if relanzar:
                    raise
                return None
            dur = time.perf_counter() - t0
            historial.append(RegistroEjecucion(f.__qualname__, inicio, dur, True,
                                               etiquetas=etiquetas))
            nivel = logging.WARNING if umbral_s is not None and dur > umbral_s else logging.INFO
            log.log(nivel, "✓ %s completado en %.3f s%s", f.__qualname__, dur,
                    " (supera umbral)" if nivel == logging.WARNING else "")
            return resultado
        return envoltura

    # Permite @monitorear (func recibido directamente) y @monitorear(...).
    return decorador(func) if callable(func) else decorador
