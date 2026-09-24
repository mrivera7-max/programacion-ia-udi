"""Pruebas automáticas (pytest) de los cuatro ejercicios del taller."""
from __future__ import annotations

import pandas as pd
import pytest

import ej4_prioridad as ej4
from ej1_poo.cargadores import CargadorDatos, ErrorCargaDatos, LimpiadorTelemetria
from ej3_mensajes.estrategias import AnalizadorMensajes
from ej3_mensajes.fabrica import FabricaEstrategias
from monitoreo import historial, monitorear


# ----------------------------------------------------------------- Ejercicio 1
@pytest.mark.parametrize("fmt", ["csv", "json", "txt", "xlsx"])
def test_cargadores_sin_nulos_y_mismo_esquema(fmt: str) -> None:
    df = CargadorDatos.desde_ruta(f"datos/telemetria.{fmt}").cargar()
    assert df.isna().sum().sum() == 0
    assert df["temperatura_c"].between(-40, 125).all()
    assert set(df["estado"]) <= {"OK", "FALLA"}


def test_limpiador_convierte_na_y_coma_decimal() -> None:
    crudo = pd.DataFrame({"id_sensor": ["S1", "S1", "S2"], "timestamp": ["t1", "t2", "t3"],
                          "temperatura_c": ["30", "31", "32"], "voltaje_v": ["5,1", "N/A", "4.9"],
                          "corriente_ma": ["100", "101", "102"], "estado": [" ok", "OK", "falla"]})
    limpio = LimpiadorTelemetria().limpiar(crudo)
    assert len(limpio) == 2 and limpio["voltaje_v"].tolist() == [5.1, 4.9]


def test_formato_no_soportado() -> None:
    with pytest.raises(ErrorCargaDatos, match="no soportado"):
        CargadorDatos.desde_ruta("x.parquet")


# ----------------------------------------------------------------- Ejercicio 2
def test_decorador_registra_exito_y_error() -> None:
    @monitorear(relanzar=False)
    def falla() -> int:
        raise ZeroDivisionError("división")

    @monitorear
    def suma(a: int, b: int) -> int:
        """Suma."""
        return a + b

    n = len(historial)
    assert suma(2, 3) == 5 and suma.__name__ == "suma" and suma.__doc__ == "Suma."
    assert falla() is None
    assert [r.exito for r in historial[n:]] == [True, False]


def test_decorador_relanza_por_defecto() -> None:
    @monitorear
    def explota() -> None:
        raise ValueError("x")

    with pytest.raises(ValueError):
        explota()


# ----------------------------------------------------------------- Ejercicio 3
def test_strategy_intercambiable() -> None:
    a = AnalizadorMensajes(FabricaEstrategias.crear("prioridad_reglas"))
    assert a.analizar_lote(["URGENTE incendio"])[0].etiqueta == "alta"
    a.estrategia = FabricaEstrategias.crear("sentimiento")
    assert a.analizar_lote(["Gracias por el apoyo"])[0].etiqueta == "positivo"


def test_fabrica_nombre_invalido() -> None:
    with pytest.raises(KeyError):
        FabricaEstrategias.crear("no_existe")


# ----------------------------------------------------------------- Ejercicio 4
@pytest.mark.parametrize(("mensaje", "esperado"), [
    ("Falla CRÍTICA en la fuente", ej4.Prioridad.ALTA),       # tildes y mayúsculas
    ("Batería al 12 %", ej4.Prioridad.ALTA),                  # regla numérica
    ("Batería al 80 %", ej4.Prioridad.BAJA),
    ("Error intermitente del equipo", ej4.Prioridad.MEDIA),
    ("Los errores de ayer ya se corrigieron", ej4.Prioridad.MEDIA),  # limitación: sin contexto
    ("Reunión el viernes", ej4.Prioridad.BAJA),
    ("Revisaremos el fallecimiento del proyecto", ej4.Prioridad.BAJA),  # evita falsos positivos
])
def test_clasificacion(mensaje: str, esperado: ej4.Prioridad) -> None:
    assert ej4.clasificar(mensaje).prioridad is esperado
