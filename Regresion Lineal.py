# -*- coding: utf-8 -*-
"""


SIMPLE LINEAR REGRESSION
"""

## Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importar el Dataset
dataset  = pd.read_csv("Salary Data.csv")

# Seleccionar la variable independiente (Years of Experience) y la dependiente (Salary)
X = dataset.iloc[:, 4].values.reshape(-1, 1)  # Years of Experience
y = dataset.iloc[:, 5].values  # Salary

# Imputar valores faltantes en 'Years of Experience' y 'Salary' utilizando la media
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)
y = imputer.fit_transform(y.reshape(-1, 1)).ravel()

# Dividir el dataset en conjunto de entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Crear y entrenar el modelo de regresión lineal
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predecir los resultados del conjunto de prueba
y_pred = regression.predict(X_test)

# Visualización de resultados del conjunto de entrenamiento
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regression.predict(X_train), color='blue')
plt.title('(Conjunto de entrenamiento)')
plt.xlabel('Años de Experiencia')
plt.ylabel('Sueldo ($)')
plt.show()

# Visualización de resultados del conjunto de prueba
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, regression.predict(X_test), color='blue')
plt.title('Valores de prueba para la regresión lineal')
plt.xlabel('Años de Experiencia')
plt.ylabel('Sueldo ($)')
plt.show()
