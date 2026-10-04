# Bitácora de IA asistida (prompts)

Asistente: Claude (Anthropic). Cada prompt indica objetivo, texto, resultado y
**validación humana** (qué se verificó antes de aceptar la salida). La IA propone;
la decisión técnica queda documentada y verificada por la autora.

| ID | Etapa | Objetivo |
|----|-------|----------|
| P1 | Comprensión | Entender el dataset y sus riesgos metodológicos |
| P2 | Ingeniería de características | Derivar variables con sentido físico |
| P3 | Refactorización | Pasar del script secuencial a POO |
| P4 | Calidad | PEP 8, docstrings y pruebas |
| P5 | Evaluación | Métrica y umbral adecuados al desbalance |
| P6 | Visualización | Dashboard orientado a decisiones |

## P1 – Comprensión del dataset
**Prompt:** "Actúa como científico de datos industrial. Te paso el encabezado del
dataset AI4I 2020 (10 000 ciclos de una fresadora). Enumera: (a) qué columnas
podrían causar fuga de información si predigo `Machine failure`, (b) el
desbalance de clases y su efecto sobre la exactitud, (c) qué partición usar."

**Resultado:** señaló TWF–RNF como fuga (son el desglose de la falla), UDI y
Product ID como identificadores, la tasa de 3,39 % y la partición estratificada.
**Validación:** se confirmó con `df[...].sum()` que la suma de modos coincide con
las fallas; se excluyeron las 7 columnas (`LEAKAGE` en `pipeline.py`).

## P2 – Variables con fundamento físico
**Prompt:** "Soy ingeniera electrónica. Con temperatura de aire y proceso [K],
velocidad [rpm], torque [N·m] y desgaste [min], propone variables derivadas con
ecuación y unidades que expliquen fallas por disipación térmica, potencia y
sobreesfuerzo. No inventes umbrales; justifica cada una."

**Resultado:** ΔT = T_proc − T_aire; P = T·2πn/60; desgaste × torque.
**Validación:** prueba unitaria de P (40 N·m, 1500 rpm = 6283 W) y estudio de
ablación: PR-AUC CV 0,736 → 0,892 al agregarlas.

## P3 – Refactorización a POO
**Prompt:** "Refactoriza este script (`legacy/script_secuencial.py`) a POO con
SOLID: separa carga, validación, características, modelos y evaluación; usa
Strategy para los algoritmos y Factory para crearlos, como en la Fase I del curso.
Mantén compatibilidad con `sklearn.Pipeline` para evitar *training-serving skew*."

**Resultado:** paquete `src/pdm_pipeline/` (7 módulos funcionales, 14 clases).
**Validación:** se revisó que el escalado y la codificación estén dentro del
`Pipeline` (se ajustan solo con datos de entrenamiento de cada pliegue) y que la
salida coincida con la del script original en el mismo modelo.

## P4 – Calidad del código
**Prompt:** "Revisa el paquete contra PEP 8 (línea ≤ 100), agrega docstrings
estilo Google con Args/Returns/Raises y escribe 5 pruebas pytest para regex,
variable de potencia, validador, umbral y fábrica."

**Resultado:** `flake8` sin advertencias; `pytest`: 5 pruebas aprobadas.

## P5 – Métrica y umbral
**Prompt:** "Con 3,4 % de positivos, ¿por qué la exactitud de 0,985 del script
original es engañosa? Propón métrica principal y un método para fijar el umbral
si una falla no detectada cuesta 10 veces una inspección."

**Resultado:** PR-AUC como métrica de selección; umbral por costo mínimo
calculado con predicciones *out-of-fold*, nunca con el conjunto de prueba.
**Validación:** umbral 0,34 (no 0,50); costo en prueba 116 vs. 680 sin modelo.

## P6 – Dashboard
**Prompt:** "Diseña un dashboard Plotly para un jefe de mantenimiento: cada
gráfico debe responder una pregunta (dónde falla, qué modelo, qué umbral, qué
vigilar). Agrega un simulador Gradio con deslizadores."

**Resultado:** `outputs/dashboard.html` (8 figuras + KPI) y `app_gradio.py`.
**Validación:** se verificó que el simulador marque alarma en 1300 rpm, 65 N·m,
215 min (95,9 %) y no en condición nominal (0 %).
