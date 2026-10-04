# -*- coding: utf-8 -*-
"""
VERSIÓN 0 (ANTES DE LA REFACTORIZACIÓN).

Script secuencial de la primera iteración del proyecto. Se conserva como
evidencia del punto de partida: todo vive en el ámbito global, no hay clases,
no hay validación de datos, el split no es estratificado ni reproducible y la
métrica reportada (exactitud) es engañosa en un problema desbalanceado
(3,39 % de fallas).

Este archivo fue el insumo del prompt P3 de la bitácora (refactorización a POO).
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/ai4i2020.csv")
X = df[["Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]]
y = df["Machine failure"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))  # ~0.98: parece excelente, no lo es
