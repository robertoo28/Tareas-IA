# -*- coding: utf-8 -*-
"""
Created on Wed Oct-16  18:14:30 2024

@author: Javier Gómez R.

MULTIPLE LINEAR REGRESSION
Restrincciones:
    - Linealidad
    - Homocedasticidad
    - Normalidad multivariable
    - Independcencia de errores
    - Ausencia de multicolinealidad (Variable Dummy - 1) -  movimiento circular entre variables
                    si hay mas de una var. Dummy hay que quitar a cada variable 1    
    Necesita satisfaces todas las restricciones para poder utilizar la RM
"""
'''
5 Métodos para construir modelos multivariable:
    **** Cada uno de ellos entra en un proceso llamado regresión paso a paso
    
    - Exhaustivo (All-in) ---> Meter todas las variables en el modelo y veamos que sale de ahí
            ** Criterios (Dominio de experto, Por necesidad->mandato del jefe,  )
    
    - Eliminación hacia atrás
            **Criterios
                    1- Seleccionar el nivel de significación para permanecer en el modelo (SL = 0.05)
                    2- Se calcula el modelo de la posibles variable predictoras
                    3- Considera la variable predictora con el p-valor más grande (Si P>SL => 4 Caso Contrario Fín)
                    4- Se elimina la variable predictora
                    5- y Se ajuste el modelo (fit) y volver a calcular (volver al paso 3)
    
    - Selección hacia adelante 
            ** Criterios
                    1- Seleccionar el nivel de significación para permanecer en el modelo (SL = 0.05)
                    2- Ajustamos todos los modelos de regresión lineal simple y~Xn, Elegimos el tiene menor P-Valor
                    3- Conservamos esta variable y ajustamos todos los posibles modelos con una variable extra añadida
                       a la o las que tengamos hasta el momento.
                    4- Consideramos la variable predictora con el menor P-Valor, (si P<SL, vamos al Paso 3, sino FIN)
                    
    - Eliminación Bidireccional (El más tedioso -- es el mejor y esta automatizado)
            ** Criterios (Combinación hacia atrás y hacia Adelante)
                    1- Seleccionar un nivel de significación para entrar y salir del modelo
                        ejm. SLENTER = 0.05, SLSTAY = 0.05
                    2- Llevar a cabo el siguiente paso de Selección hacia adelante (con las nuevas variables con P<SLENTER
                                                                                    para entrar)
                    3- Llevar todos los pasos de la eliminación hacia atrás (Las variables antiguas dbeen tener P<SLSTAY
                                                                             para quedarse)
                    4- No hay variables para entrar ni antiguas para salir - FIN
    - Comparación de Scores
            ** Criterios
                    1- Seleccionar un criterio de bondad de ajuste (Criterio de Akaike)
                    2- Construir todos los posibles modelos de regresión 2ExpN - 1 combinaciones en total
                    3- Seleccionar el modelo con el mejor criterio elegido
                    4- Fin
                Ejemplo: con 10 varibles (Columnas de datos)--> 1023 modelos
    
'''
## Análisis de empresas Statups -- decisión de inversión de acuerdo al tipo que querramos
## Vamos a predecir en cuál hay que invertir de acuerdo a algún criterio
## Tomar encuenta si hay algúna correlación entre las ganancias y la ubicación?
## Predecir cuanto gana la empresa (Y)

## Importar las librearías
import numpy as np ## Libraría para tratamiento de números y herramientas matemáticas (np Alias)
import matplotlib.pyplot as plt  ## Librería para representación gráfica (pyplot es una sublibrearía - gráficos) (plt Alias)
import pandas as pd  ## Librería para manipular datos (pd Alias)


## Importar el Dataset
dataset  = pd.read_csv("non_linear_regression_dataset.csv")

## Saber con que filas y columnas se trabajará de la Var. Predictora
## Variables independientes "X" tiene 3 Col y 10 Fil
## El menos 1 es la última columna y es era para predecir

X = dataset.iloc[:,:-1].values ## iloc Localizar elementos de filas y columnas por posición (i - Index -- loc Localization)
## Los primeros : puntos desde la fila inicio hasta la fila fin
## Los dos siguientes : puntos desde  la columna inicio hasta el fin
## El negativo es excepto la últimva
## .values solo los valores del arreglo

## Obtener datos de la columna a Predecir
y = dataset.iloc[:, 3].values ## Los dos primeros puntos : filas y los dos punto siguiente columnas

'''

## Tratamiento de los NA's (qué es un NAN)
## Libraría para limipieza e imputación de datos
## from sklearn.preprocessing import Imputer línea del código original con Imputer no funciona
### Original from sklearn.preprocesssing import Imputer
from sklearn.impute import SimpleImputer

## Creación de un objeto de la clase Imputer
## Que vamos a buscar y que estratégia utilizaremos
## axis = 0 aplicando la media por columna

imputer = SimpleImputer(missing_values = np.nan, strategy = "mean") ## Estratégia reemplazar por la media de la columna
##imputer.fit(X)  ## Con esto es la media a todas las columnas
## Seleccionar los NAN
imputer = imputer.fit(X[:, 1:3]) ## tomamos la columnas de la 1 a la 2 ya que es n-1 (3-1=2) -- ojo si se pone solo Fit de X
## Colocar los valores de la media de NAN
X[:, 1:3] = imputer.transform(X[:, 1:3]) ## Ejecuta la ejecución de la imputación Todas las filas y solo 2 columnas edad y sueldo



## Codificar datos categóricos

## from sklearn import preprocessing 
## Primero la variable categógica de las variables independientes
labelencoder_X = LabelEncoder();
## labelencoder_X.fit_transform(X[:, 0]) -> Matriz de valores con 0, 1, 2, etc
## Estas son variables Dummy - La Idea es un One Hot Encode en 1 sola columna
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) ## Codifica y convierte a datos numéricos

'''

from sklearn.preprocessing import OneHotEncoder, LabelEncoder  ## Llamamos a las Variables Dummy
from sklearn.compose import ColumnTransformer

#labelencoder_X = LabelEncoder();
## labelencoder_X.fit_transform(X[:, 0]) -> Matriz de valores con 0, 1, 2, etc
#X[:, 2] = labelencoder_X.fit_transform(X[:, 2]) ## Codifica y convierte a datos numéricos

## Primero a número y luego a variable Dummy --> OneHotEncoder or Vector -> Traducir una categoría que no tiene orden
## a un conjunto de tantas columnas como categorías existen
#onehotencoder = ColumnTransformer([('one_hot_encoder',OneHotEncoder(categories='auto'), [3])], remainder='passthrough')

## variable Dummy para variable X
#X = np.array(onehotencoder.fit_transform(X),dtype=float) ## Genera una matriz de características

'''
## para variable Dicotómica - Solo es necesario un Label Encoder y no One Hot Encoder - BooL Value 
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

'''
################-------------------------------
####### Esto normalmente quedaría en la plantilla principal

'''
Evitar el tema de las variables ficticias

'''
#X = X[:,1:] ## Eliminando la primera columna (Ojo que hay algunos métodos quitan variables ficticias)



## Dividir el Dataset en conjunto entrenamiento y conjunto testing
from sklearn.model_selection import train_test_split
## Esto devolverá 4 variables y hay que crearlas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 
## Random es para que siempre tome los mismos datos

'''
## Escalar los datos -> Mostrar lámina de Standarización y Normalización
## Para que no exista outloyers y existes 2 formas
## Xstan = (x- mean(x))/standar deviation (Standarización) -- en Relación a la media
## Xnorm 0 (x - min(x))/(max(x)-min(x))   (Normalización)  -- Pequeño 0 mas Gránde 1 y el resto escalado de forma lineal

### Escalado de variables
### Procesamos la estandarización
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test  = sc_X.transform(X_test) ## el conjunto X de test se escalará con la misma transformación de los de entrenamiento.


## En el caso de la Regresión Lineal Simple -- normalmente no rqeuiere Escalado
## ESCALADO -> Normalización o Estandarización
'''


''' MODELO DE REGRESION MULTIPLE'''
### Creación del modelo de regresión MÚLTIPLE con los datos de entrenamiento
### Ajuste del modelo
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)  ## Importante tener el nmismo número de filas y aprendió lo que tenía como dato de entrenamiento

### Predecir el modelo y probarlo con test
y_pred = regression.predict(X_test)


### Añadir o Quitar variables - Utilizaremos eliminación hacia atrás
import statsmodels.api as sm 
## Calcular el P_Valor
X = np.append(arr = np.ones((100,1)).astype(int), values = X , axis=1)
## Creación del nivel de significación
SL = 0.05

# Asegurarse de que no haya valores NaN
#imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
#X = imputer.fit_transform(X)

# Asegurarse de que todos los datos sean numéricos (float)
#X = X.astype(float)

### Matriz de características óptimas
X_opt = X[:,[0,1,2]]  ## Tomar todas las columnas
## Creear una nueva regresión de las nuevas variables
regression_OLS = sm.OLS(y, X_opt).fit()
regression_OLS.summary()

### Matriz de características óptimas - ajustado
X_opt = X[:,[0,2]]  ## Tomar todas las columnas
## Creear una nueva regresión de las nuevas variables
regression_OLS = sm.OLS(y, X_opt).fit()
regression_OLS.summary()


### Matriz de características óptimas - ajustado
#X_opt = X[:,[0,3,4,5]]  ## Tomar todas las columnas
## Creear una nueva regresión de las nuevas variables
#regression_OLS = sm.OLS(y, X_opt).fit()
#regression_OLS.summary()

### Matriz de características óptimas - ajustado
#X_opt = X[:,[0,3,5]]  ## Tomar todas las columnas
## Creear una nueva regresión de las nuevas variables
#regression_OLS = sm.OLS(y, X_opt).fit()
#regression_OLS.summary()

### Matriz de características óptimas - ajustado
#X_opt = X[:,[0,3]]  ## Tomar todas las columnas
## Creear una nueva regresión de las nuevas variables
#regression_OLS = sm.OLS(y, X_opt).fit()
#regression_OLS.summary()



## Visualización de resultados con el Entrenamiento
#plt.scatter(X_train, y_train, color = "red")
#plt.plot(X_train, regression.predict(X_train), color = "blue")
#plt.title("Sueldo vs. Años de Experiencia (Conjunto de entrenamiento)")
#plt.xlabel("Años de Experiencia")
#plt.ylabel("Sueldo ($)")
#plt.show()

## Visualización de resultados con el test
#plt.scatter(X_test, y_test, color = "red")
#plt.plot(X_test, regression.predict(X_test), color = "blue")
#plt.title("Sueldo vs. Años de Experiencia (Conjunto de test)")
#plt.xlabel("Años de Experiencia")
#plt.ylabel("Sueldo ($)")
#plt.show()















