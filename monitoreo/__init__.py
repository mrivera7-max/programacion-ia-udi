"""Paquete de monitoreo transversal para procesos de entrenamiento de IA.

Expone únicamente la API pública; los detalles internos quedan en submódulos.
"""
from monitoreo.decoradores import RegistroEjecucion, historial, monitorear

__all__ = ["monitorear", "historial", "RegistroEjecucion"]
__version__ = "1.0.0"
