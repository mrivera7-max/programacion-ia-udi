# Taller Fase I — Programación con IA
### POO, decoradores y patrones de diseño asistidos por IA en Python

Actividad colaborativa (Fase I, individual) del curso **Programación con IA**, Unidades 2 y 3.
Maestría en Ciencia de Datos e Inteligencia Artificial — **Universidad de Investigación y Desarrollo (UDI)**.

**Autora:** María Fernanda Rivera Sanclemente · Bucaramanga, Colombia · Septiembre de 2026

---

## Descripción

Caso de estudio único: **telemetría de sensores y gestión de incidentes en un laboratorio de
electrónica**. Sobre él se desarrollan los cuatro ejercicios del taller, cada uno siguiendo el mismo
ciclo metodológico: *problema → prompt estructurado → generación asistida por IA → verificación
objetiva → refinamiento crítico*.

| Ejercicio | Tema | Entregable en este repositorio |
|---|---|---|
| 1 | Refactorización de un proceso estructurado de carga y limpieza a **POO** (herencia, polimorfismo, Template Method, registro por decorador) | `ej1_poo/` |
| 2 | **Decorador** `@monitorear` (tiempo de ejecución y excepciones) empaquetado como módulo independiente | `monitoreo/`, `ej2_entrenamiento.py` |
| 3 | Patrones **Factory** y **Strategy** para un sistema flexible de análisis de mensajes | `ej3_mensajes/` |
| 4 | Clasificación de prioridad con **expresiones regulares**: IA simbólica vs. scikit-learn | `ej4_prioridad.py` |

El informe completo (justificaciones técnicas, auditoría de la respuesta inicial de la IA y
conclusiones) está en [`docs/`](docs/).

## Hallazgos principales

- La primera respuesta de la IA en el Ejercicio 1 ordenaba mal la limpieza (`dropna` antes de
  `to_numeric`). El defecto **solo se manifiesta con JSON** (480 filas con 10 nulos en lugar de 470),
  porque `read_csv` interpreta «N/A» como nulo por defecto: un error dependiente del formato que una
  prueba únicamente con CSV habría aprobado.
- El diseño modular **no penaliza el rendimiento**: con 200 000 filas, 0,58 s (POO) frente a 0,76 s
  (script estructurado).
- Con 30 mensajes etiquetados, las reglas simbólicas alcanzan **96,7 %** de exactitud frente a
  **56,7 %** de TF-IDF + regresión logística con validación cruzada de 5 pliegues.

## Instalación

```bash
git clone https://github.com/mrivera7-max/programacion-ia-udi.git
cd programacion-ia-udi
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requiere Python 3.12 o superior.

## Ejecución

```bash
python generar_datos.py                    # 1. genera datos/ (telemetría + mensajes)
python ej1_poo/procedural_original.py      # Ej. 1: línea base estructurada
python ej1_poo/sugerencia_inicial_ia.py    # Ej. 1: respuesta inicial de la IA (auditada)
python ej1_poo/cargadores.py               # Ej. 1: versión POO corregida
python ej1_poo/validar_ej1.py              # Ej. 1: equivalencia, eficiencia, extensibilidad
python ej2_entrenamiento.py                # Ej. 2: decorador @monitorear en entrenamiento
python ej3_mensajes/demo_ej3.py            # Ej. 3: Factory + Strategy
python ej4_prioridad.py datos/mensajes --etiquetas datos/mensajes_etiquetados.csv --comparar-ml
```

Calidad del código:

```bash
python -m pytest -q tests   # 17 pruebas
ruff check .                # estilo (PEP 8) y defectos comunes
mypy ej1_poo/cargadores.py monitoreo ej3_mensajes ej4_prioridad.py ej2_entrenamiento.py
```

## Estructura

```
ej1_poo/                 Ej. 1: original estructurado, prompt, sugerencia de la IA,
                         versión POO corregida y script de validación
monitoreo/               Ej. 2: paquete con el decorador @monitorear
ej2_entrenamiento.py     Ej. 2: script principal (entrenamiento monitoreado)
ej3_mensajes/            Ej. 3: estrategias (Strategy) y fábrica (Factory)
ej4_prioridad.py         Ej. 4: clasificador de prioridad por expresiones regulares
generar_datos.py         Generación reproducible de los datos (semilla fija)
tests/                   17 pruebas automáticas (pytest)
salidas/                 Salidas de ejecución citadas en el informe
docs/                    Informe del taller
.github/workflows/       Integración continua: pruebas, ruff y mypy
```

## Reproducibilidad

Los datos son sintéticos y se generan con semilla fija (`numpy.random.default_rng(42)`).
Entorno de referencia: Python 3.12.3, pandas 3.0.2, scikit-learn 1.8.0.

## Uso de IA

Se utilizó un asistente de IA generativa como herramienta de apoyo para generar propuestas de
código, conforme a lo solicitado en los ejercicios del curso. Todo el código fue ejecutado, probado
y auditado; los hallazgos, correcciones y decisiones de diseño se documentan en el informe.

## Licencia

Distribuido bajo licencia MIT. Ver [LICENSE](LICENSE).
