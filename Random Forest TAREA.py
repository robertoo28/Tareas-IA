### Algoritmo de Regresión Logística. 

# Se da a variables Cualitativas categoricas. 
# 
# A quien se aproxima más, de eso trata este algoritmo.
'''
Random Forest.

Modelo de clasificación 

Pasos: 
    1. Seleccionar un número de datos del conjunto de entrenamiento
    2. Se construye el arbol en base a esos datos que pase 
    3. Elegimos un número de arboles que van a construir y repetimos los pasos anteriores
    4. 
    
    Leer Real time Human Reconigtion in parts from single Depth Images 
    
    Por default, python toma 10 en la generación de random forest 
    El algoritmo tiende mas al sobre ajuste 
    Para darnos cuenta, dentro del gráfico veremos que hay varias areas 
    

'''

## Importar Librerias ## 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

##Leer CSV
dataset = pd.read_csv("Social_Dataset.csv")

## Matriz de acaracteristicas 
X = dataset.iloc[: , [2,3]].values # Usando las columnas Credit_Score y Loan_Amount

y = dataset.iloc[:, -1].values  # Usando Loan_Approval como objetivo

## Dividir el dataset en entranamiento y test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

## Escalar variables 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

## Modelo de regresión logistica al conjunto de entrenamiento 
## Aprende a predecir clasificaciones 
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, criterion= "entropy", random_state=0) # Entropia, indice Giny es el por defecto
rf.fit(X_train, y_train)

## Prediccion de los resultados en el conjunto de test
y_pred = rf.predict(X_test)

## Resultados en función de la matriz de confusión
# Se calcula la matriz de confusión en basea 
from sklearn.metrics import confusion_matrix
conmet = confusion_matrix(y_test, y_pred)

##Visualizar el algoritmo 
### Visualizar el algotirmo de train graficamente con los resultados
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_train and y_train are defined
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
)

# Define a colormap with standard color names or hex codes
cmap = ListedColormap(['red', 'green'])  # Or use hex codes ['#7f00ff', '#00ff7f']

# Plot decision boundary
plt.contourf(X1, X2, rf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=cmap)

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# Plot training points
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                color=cmap(i), label=j)  # Using the colormap to assign colors

plt.title("Random Forest (Training set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

### Visualizar el algotirmo de test graficamente con los resultados
# Assuming X_test and y_test are defined
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
)

# Define a colormap with standard color names or hex codes
cmap = ListedColormap(['red', 'green'])  # Red and green hex codes

# Plot decision boundary
plt.contourf(X1, X2, rf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=cmap)

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# Plot test points
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                color=cmap(i), label=j)  # Using the colormap to assign colors


plt.title("Random Forest (Test set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()
