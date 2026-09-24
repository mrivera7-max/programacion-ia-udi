"""Ejercicio 4: clasificación de prioridad de mensajes con expresiones regulares.

Lee todos los archivos .txt de una carpeta (un mensaje por línea), detecta
palabras clave con regex y clasifica cada mensaje como ALTA, MEDIA o BAJA.
Enfoque: IA simbólica (sistema basado en reglas), justificado en el informe y
contrastado empíricamente con un modelo de scikit-learn (opción --comparar-ml).

Uso:
    python ej4_prioridad.py datos/mensajes --salida resultados_prioridad.csv \
        --etiquetas datos/mensajes_etiquetados.csv --comparar-ml
"""
from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class Prioridad(IntEnum):
    """Niveles ordenados: permite comparar (ALTA > MEDIA > BAJA)."""

    BAJA = 1
    MEDIA = 2
    ALTA = 3


# Reglas declarativas: el conocimiento experto vive en DATOS, no en if/elif.
# Los patrones se aplican sobre texto normalizado (minúsculas y sin tildes).
REGLAS: dict[Prioridad, list[str]] = {
    Prioridad.ALTA: [
        r"\burgente\b", r"\bcritic[oa]s?\b", r"\bincendio\b", r"\bhumo\b",
        r"\bchispas?\b", r"\bquemad[oa]\b", r"\bcortocircuito\b",
        r"\bdescarga electrica\b", r"\bpeligro\b", r"\briesgo electrico\b",
        r"\bca[ií]d[oa]\b", r"\bno responde\b", r"\bperdida de datos\b",
        r"\bacceso no autorizado\b", r"\bde inmediato\b", r"\bpelad[oa]\b",
    ],
    Prioridad.MEDIA: [
        r"\bfall[aoó]\w*\b", r"\berror\w*\b", r"\brevisar\b", r"\blent[oa]\b",
        r"\bno enciende\b", r"\bintermitente\b", r"\bretraso\b",
        r"\bmantenimiento\b", r"\bactualizar\b", r"\binestables?\b",
    ],
}
# Regla numérica (algo que una lista de palabras no captura): batería <= 15 %.
PATRON_BATERIA = re.compile(r"\bbateria\b.*?\b(\d{1,3})\s*%")
UMBRAL_BATERIA = 15

_COMPILADAS = {p: [re.compile(x) for x in pats] for p, pats in REGLAS.items()}


def normalizar(texto: str) -> str:
    """Minúsculas y sin tildes, para que 'Crítica' y 'critica' coincidan."""
    nfkd = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


@dataclass(frozen=True)
class Clasificacion:
    """Resultado explicable: prioridad + evidencia que la justifica."""

    mensaje: str
    prioridad: Prioridad
    evidencias: tuple[str, ...]


def clasificar(mensaje: str) -> Clasificacion:
    """Asigna la prioridad más alta cuyas reglas se activen (BAJA por defecto).

    Args:
        mensaje: Texto del mensaje original.

    Returns:
        Clasificación con las coincidencias que la originaron (trazabilidad).
    """
    texto = normalizar(mensaje)
    bat = PATRON_BATERIA.search(texto)
    if bat and int(bat.group(1)) <= UMBRAL_BATERIA:
        return Clasificacion(mensaje, Prioridad.ALTA, (f"bateria {bat.group(1)}%",))
    for nivel in (Prioridad.ALTA, Prioridad.MEDIA):          # de mayor a menor
        hallazgos = tuple(m.group(0) for rx in _COMPILADAS[nivel] if (m := rx.search(texto)))
        if hallazgos:
            return Clasificacion(mensaje, nivel, hallazgos)
    return Clasificacion(mensaje, Prioridad.BAJA, ())


def leer_mensajes(carpeta: Path) -> Iterator[tuple[str, str]]:
    """Genera (archivo, mensaje) de cada línea no vacía de cada .txt (lectura perezosa)."""
    archivos = sorted(carpeta.glob("*.txt"))
    if not archivos:
        raise FileNotFoundError(f"No hay archivos .txt en {carpeta}")
    for ruta in archivos:
        try:
            with ruta.open(encoding="utf-8") as f:
                for linea in f:
                    if linea.strip():
                        yield ruta.name, linea.strip()
        except UnicodeDecodeError:
            logger.warning("Se omite %s: no está codificado en UTF-8", ruta.name)


def comparar_con_ml(etiquetado: pd.DataFrame) -> None:
    """Evalúa un modelo scikit-learn (TF-IDF + regresión logística) con
    validación cruzada estratificada, para contrastarlo con las reglas."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline

    modelo = make_pipeline(TfidfVectorizer(preprocessor=normalizar, ngram_range=(1, 2)),
                           LogisticRegression(max_iter=1000))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pred = cross_val_predict(modelo, etiquetado["mensaje"], etiquetado["prioridad_real"], cv=cv)
    exactitud = (pred == etiquetado["prioridad_real"]).mean()
    print(f"\nModelo scikit-learn (TF-IDF + LogReg, CV 5 pliegues, n={len(etiquetado)}): "
          f"exactitud = {exactitud:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clasifica mensajes .txt por prioridad.")
    parser.add_argument("carpeta", type=Path)
    parser.add_argument("--salida", type=Path, default=Path("resultados_prioridad.csv"))
    parser.add_argument("--etiquetas", type=Path, help="CSV con prioridad_real para evaluar")
    parser.add_argument("--comparar-ml", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    filas = [{"archivo": a, "mensaje": c.mensaje, "prioridad": c.prioridad.name.lower(),
              "evidencias": ", ".join(c.evidencias)}
             for a, m in leer_mensajes(args.carpeta) for c in [clasificar(m)]]
    df = pd.DataFrame(filas)
    df.to_csv(args.salida, index=False, encoding="utf-8")
    print(df.assign(mensaje=df["mensaje"].str.slice(0, 48))
            [["mensaje", "prioridad", "evidencias"]].to_string(index=False))
    print(f"\nDistribución: {df['prioridad'].value_counts().to_dict()} -> {args.salida}")

    if args.etiquetas and args.etiquetas.exists():
        from sklearn.metrics import confusion_matrix
        real = pd.read_csv(args.etiquetas)
        comp = df.merge(real, on=["archivo", "mensaje"])
        exactitud = (comp["prioridad"] == comp["prioridad_real"]).mean()
        orden = ["alta", "media", "baja"]
        print(f"\nReglas (IA simbólica), n={len(comp)}: exactitud = {exactitud:.1%}")
        print(pd.DataFrame(confusion_matrix(comp["prioridad_real"], comp["prioridad"],
                                            labels=orden),
                           index=[f"real_{o}" for o in orden],
                           columns=[f"pred_{o}" for o in orden]).to_string())
        errores = comp[comp["prioridad"] != comp["prioridad_real"]]
        for _, e in errores.iterrows():
            print(f"  Error: '{e['mensaje'][:55]}' real={e['prioridad_real']} "
                  f"pred={e['prioridad']}")
        if args.comparar_ml:
            comparar_con_ml(real)


if __name__ == "__main__":
    main()
