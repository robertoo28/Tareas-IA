### Regresión logistica. 


## Habla de probabilidad, por ejemplo probabilidad de adquirir un vehiculo. 
# Se da a variables cuantitativas. 
# 
'''
LA regresión logistica es un algoritmo de clasificación. 
Responde a preguntes en base a una función sigmoide. 

La curva de la regresión logistica está dada por      
(Sigmoide )ln ( p / 1 - p )   = b0 + b1 x  (Recta) 

Siempre tiene un elemento de clasfiicación. 
 

Las variables para que se puedan compara se les debe escalar 

'''


## Importar Librerias ## 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

##Leer CSV
dataset = pd.read_csv("Social_Network_Ads.csv")

## Matriz de acaracteristicas 

X = dataset.iloc[: , [2,3]]


y = dataset.iloc[:, -1]




    


## Dividir el dataset en entranamiento y test



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)




## Escalar variables 


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


## Modelo de regresión logistica al conjunto de entrenamiento 
## Aprende a predecir clasfiicaciones 



from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)



## Prediccion de los resultados en el conjunto de test

y_pred = classifier.predict(X_test)


## Resultados en función de la matriz de confusión
# Se calcula la matriz de confusión en basea 

## Verificar preddiciones correctas 


from sklearn.metrics import confusion_matrix
conmet = confusion_matrix(y_test, y_pred)


# 0 - 0 65 no van a comprar nada 1 - 1 24 van a comprar sí o sí


#  0 - 1 y 1 - 0 son error 

# Para calcular los porcentajes de error. 65 + 24 y 8 + 3. 89% de bien 11% de error. 




##Visualizar el algoritmo 
### Visualizar el algotirmo de train graficamente con los resultados
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2 , classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha = 0.75, cmap = ListedColormap(('red', 'green')))                     
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title("K-NN (Training set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()
 
 
### Visualizar el algotirmo de test graficamente con los resultados
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2 , classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha = 0.75, cmap = ListedColormap(('red', 'green')))                     
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title("K-NN (Test set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()



## EL QUE FUNCIONA ES EL DE KNN POR SI ACASO, COPIAR CÓdigo de AHI










