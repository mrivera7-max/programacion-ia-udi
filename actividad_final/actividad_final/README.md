# Actividad Final – Pipeline de IA para mantenimiento predictivo

Maestría en Ciencia de Datos e IA (UDI) · Programación con IA
Autora: María Fernanda Rivera Sanclemente · Tema: **Automatización de procesos**

Pipeline en Python, orientado a objetos, que predice fallas de una fresadora CNC
a partir de 6 variables de sensor (dataset AI4I 2020, 10 000 ciclos, 3,39 % de fallas).

## Resultados (conjunto de prueba, n = 2000)

| Métrica | Valor |
|---|---|
| Modelo elegido (CV) | Random Forest |
| PR-AUC / ROC-AUC | 0,878 / 0,983 |
| Recall / Precisión | 83,8 % / 90,5 % |
| Umbral por costo (FN = 10, FP = 1) | 0,34 |
| Costo vs. no predecir | 116 vs. 680 (−83 %) |
| Ablación PR-AUC CV (sin → con variables físicas) | 0,736 → 0,892 |

## Estructura

```
actividad_final/
├── data/ai4i2020.csv              # dataset (se descarga de UCI si falta)
├── legacy/script_secuencial.py    # versión 0, antes de refactorizar
├── src/pdm_pipeline/
│   ├── config.py                  # PipelineConfig (dataclass)
│   ├── data.py                    # DataLoader (regex) + DataValidator
│   ├── features.py                # PhysicsFeatureEngineer (ΔT, potencia, esfuerzo)
│   ├── models.py                  # Strategy + Factory (LR, RF, HGB)
│   ├── evaluation.py              # Evaluator + ThresholdOptimizer por costo
│   ├── pipeline.py                # Facade: orquesta todo + logging
│   ├── dashboard.py               # Dashboard Plotly (HTML interactivo)
│   └── static_figures.py          # Figuras PNG para el informe
├── app_gradio.py                  # Simulador "¿qué pasa si?" (Gradio)
├── notebooks/Actividad_Final_Pipeline_IA.ipynb   # ejecución en Colab
├── prompts/bitacora_prompts.md    # evidencia de IA asistida
└── tests/test_pipeline.py         # 5 pruebas pytest
```

## Ejecución

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pdm_pipeline   # entrena, evalúa y genera outputs/
python app_gradio.py                    # simulador interactivo
pytest -q tests                         # pruebas
```

Salidas en `outputs/`: `modelo_pdm.joblib`, `metricas.json`,
`predicciones_prueba.csv`, `dashboard.html`, `figuras/*.png`.

Calidad: PEP 8 verificado con `flake8 --max-line-length 100`; docstrings estilo Google.

## Dataset

Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset* [Conjunto de datos].
UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C
