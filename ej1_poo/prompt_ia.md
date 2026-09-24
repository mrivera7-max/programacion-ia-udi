# Prompt enviado al asistente de IA (Ejercicio 1)

**Rol:** Actúa como ingeniero de software senior especializado en Python 3.12,
pandas y diseño orientado a objetos (SOLID).

**Contexto:** Tengo un script estructurado (`cargar_y_limpiar(ruta)`) que carga
telemetría de sensores (columnas: id_sensor, timestamp, temperatura_c,
voltaje_v, corriente_ma, estado) desde archivos .csv, .json, .txt (tabulado)
y .xlsx, usando una cadena if/elif por extensión, y luego limpia los datos:
elimina duplicados, convierte comas decimales a punto, fuerza tipos numéricos,
normaliza `estado` (strip + mayúsculas), filtra temperatura fuera de
(-40, 125) °C y elimina nulos. [Se adjuntó el código original completo].

**Tarea:** Refactoriza el script a Programación Orientada a Objetos:
1. Una clase base abstracta `CargadorDatos` (módulo `abc`) que defina la
   interfaz común y la lógica de limpieza compartida.
2. Una subclase por formato (CSV, JSON, TXT, XLSX) que solo implemente la
   lectura, aprovechando herencia y polimorfismo.
3. Un mecanismo para obtener el cargador adecuado según la extensión.

**Restricciones:**
- Python 3.12, solo pandas y la biblioteca estándar.
- Tipado explícito (typing), docstrings estilo Google, PEP 8.
- No usar `print` para errores: usar excepciones propias y `logging`.
- El resultado debe ser idéntico al del script original para los mismos datos.

**Formato de salida:** un único módulo Python y un bloque `if __name__ ==
"__main__":` que demuestre la carga polimórfica de los cuatro formatos.
