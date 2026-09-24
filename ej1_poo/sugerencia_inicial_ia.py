"""Primera respuesta del asistente de IA al prompt (registrada SIN cambios).

Se conserva como evidencia para el análisis crítico del Ejercicio 1.
Los comentarios con prefijo `# [AUDITORÍA]` fueron agregados después por la
autora para señalar los problemas detectados durante la validación.
"""
from abc import ABC, abstractmethod

import pandas as pd


class CargadorDatos(ABC):
    """Clase base para cargadores de datos."""

    def __init__(self, ruta: str):
        self.ruta = ruta

    @abstractmethod
    def leer(self) -> pd.DataFrame:
        pass

    def limpiar(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates()
        df = df.dropna()  # [AUDITORÍA] ERROR 1: dropna ANTES de to_numeric.
        for col in ["temperatura_c", "voltaje_v", "corriente_ma"]:
            df[col] = df[col].astype(str).str.replace(",", ".")
            # [AUDITORÍA] "N/A" se vuelve NaN aquí, después del dropna -> nulos filtrados.
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["estado"] = df["estado"].str.strip().str.upper()
        df = df[(df["temperatura_c"] > -40) & (df["temperatura_c"] < 125)]
        return df.reset_index(drop=True)

    def cargar(self) -> pd.DataFrame:
        try:
            return self.limpiar(self.leer())
        except Exception as e:  # [AUDITORÍA] ERROR 2: silencia cualquier fallo
            print(f"Error: {e}")
            return pd.DataFrame()  # y devuelve un DataFrame vacío "válido".


class CargadorCSV(CargadorDatos):
    def leer(self) -> pd.DataFrame:
        return pd.read_csv(self.ruta)


class CargadorJSON(CargadorDatos):
    def leer(self) -> pd.DataFrame:
        # [AUDITORÍA] ERROR 3: read_json infiere tipos distintos (timestamp ->
        # datetime64) y rompe la equivalencia de esquema entre subclases.
        return pd.read_json(self.ruta)


class CargadorTXT(CargadorDatos):
    def leer(self) -> pd.DataFrame:
        return pd.read_csv(self.ruta, sep="\t")


class CargadorExcel(CargadorDatos):
    def leer(self) -> pd.DataFrame:
        return pd.read_excel(self.ruta)


def obtener_cargador(ruta: str) -> CargadorDatos:
    # [AUDITORÍA] ERROR 4: el if/elif del script original solo se trasladó
    # aquí. Agregar un formato sigue obligando a modificar código existente (viola OCP).
    if ruta.endswith(".csv"):
        return CargadorCSV(ruta)
    elif ruta.endswith(".json"):
        return CargadorJSON(ruta)
    elif ruta.endswith(".txt"):
        return CargadorTXT(ruta)
    elif ruta.endswith(".xlsx"):
        return CargadorExcel(ruta)
    else:
        raise ValueError("Formato no soportado")


if __name__ == "__main__":
    for r in ["datos/telemetria.csv", "datos/telemetria.json",
              "datos/telemetria.txt", "datos/telemetria.xlsx"]:
        df = obtener_cargador(r).cargar()
        print(r, df.shape, "nulos:", int(df.isna().sum().sum()),
              "tipo timestamp:", df["timestamp"].dtype)
