# -*- coding: utf-8 -*-
"""Paquete del pipeline de mantenimiento predictivo (AI4I 2020).

Uso rápido::

    from pdm_pipeline import PredictiveMaintenancePipeline
    resumen = PredictiveMaintenancePipeline().run()
"""
from .config import PipelineConfig
from .pipeline import PredictiveMaintenancePipeline

__all__ = ["PipelineConfig", "PredictiveMaintenancePipeline"]
__version__ = "1.0.0"
