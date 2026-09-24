"""Patrón Strategy: familia intercambiable de algoritmos de análisis de mensajes.

Todas las estrategias cumplen el mismo contrato (``analizar``), de modo que el
contexto ``AnalizadorMensajes`` puede cambiar de algoritmo en tiempo de
ejecución sin modificar su propio código.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field

import ej4_prioridad as reglas_prioridad
from ej4_prioridad import normalizar


@dataclass(frozen=True)
class Resultado:
    """Salida uniforme de cualquier estrategia."""

    estrategia: str
    etiqueta: str
    confianza: float
    detalle: str = ""


class EstrategiaAnalisis(ABC):
    """Contrato común (interfaz Strategy)."""

    nombre: str = "base"

    @abstractmethod
    def analizar(self, mensaje: str) -> Resultado:
        """Analiza un mensaje y devuelve un ``Resultado``."""


class EstrategiaPrioridadReglas(EstrategiaAnalisis):
    """Prioridad por reglas regex (reutiliza el Ejercicio 4: DRY)."""

    nombre = "prioridad_reglas"

    def analizar(self, mensaje: str) -> Resultado:
        c = reglas_prioridad.clasificar(mensaje)
        return Resultado(self.nombre, c.prioridad.name.lower(), 1.0, ", ".join(c.evidencias))


@dataclass
class EstrategiaSentimientoLexico(EstrategiaAnalisis):
    """Tono del mensaje (positivo/negativo/neutro) con un léxico configurable."""

    positivas: frozenset[str] = frozenset({"gracias", "felicitaciones", "agradecimiento",
                                           "cordial", "apoyo", "funciona"})
    negativas: frozenset[str] = frozenset({"falla", "error", "peligro", "quemado", "caido",
                                           "lento", "lenta", "incendio", "riesgo", "no"})
    nombre: str = field(default="sentimiento", init=False)

    def analizar(self, mensaje: str) -> Resultado:
        tokens = re.findall(r"\w+", normalizar(mensaje))
        pos = sum(t in self.positivas for t in tokens)
        neg = sum(t in self.negativas for t in tokens)
        total = pos + neg
        etiqueta = "neutro" if pos == neg else ("positivo" if pos > neg else "negativo")
        return Resultado(self.nombre, etiqueta, abs(pos - neg) / total if total else 0.0,
                         f"+{pos}/-{neg}")


class EstrategiaModeloML(EstrategiaAnalisis):
    """Prioridad con un modelo scikit-learn entrenado al construirse."""

    nombre = "prioridad_ml"

    def __init__(self, textos: Iterable[str], etiquetas: Iterable[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        self._modelo = make_pipeline(TfidfVectorizer(preprocessor=normalizar),
                                     LogisticRegression(max_iter=1000))
        self._modelo.fit(list(textos), list(etiquetas))

    def analizar(self, mensaje: str) -> Resultado:
        proba = self._modelo.predict_proba([mensaje])[0]
        i = int(proba.argmax())
        return Resultado(self.nombre, str(self._modelo.classes_[i]), float(proba[i]))


class AnalizadorMensajes:
    """Contexto del patrón Strategy: delega el análisis en la estrategia actual."""

    def __init__(self, estrategia: EstrategiaAnalisis) -> None:
        self._estrategia = estrategia

    @property
    def estrategia(self) -> EstrategiaAnalisis:
        return self._estrategia

    @estrategia.setter
    def estrategia(self, nueva: EstrategiaAnalisis) -> None:
        """Permite cambiar el algoritmo en tiempo de ejecución."""
        self._estrategia = nueva

    def analizar_lote(self, mensajes: Iterable[str]) -> list[Resultado]:
        return [self._estrategia.analizar(m) for m in mensajes]
