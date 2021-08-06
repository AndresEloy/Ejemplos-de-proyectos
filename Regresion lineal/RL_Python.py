import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

#Dado a que el csv esta separado por ; los encabezados y los decimales por , es por eso que se pone esta linea para el csv
data = pd.read_csv("prostate_data.csv", sep=";",decimal=',', index_col=0)
#Separamos lpsa de los demas datos 
X = np.array(data.drop(['lpsa'], 1))
y = np.array(data['lpsa'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(data.head())
print('Numero de datos para entrenar: {} ; Numero de datos para pruebas: {}'.format(X_train.shape[0], X_test.shape[0]))
#Algoritmo de regresion lineal 
algoritmo = LinearRegression()
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)
print('Precisión Regresión Lineal: {}'.format(algoritmo.score(X_train, y_train)))
# Modelo de regresion lineal 
print("Intercept:", algoritmo.intercept_)
print("Coeficiente:", algoritmo.coef_)
print("R^2:", algoritmo.score(X, y)) #Coeficiente de determinación 
rmse = mean_squared_error(
        y_true  = y_test,
        y_pred  = Y_pred,
        squared = False
       )
print("")
print(f"RMSE: {rmse}") #error del test
#Se realiza la matriz de confusion
cutoff = 0.7                              
y_pred_classes = np.zeros_like(Y_pred)    
y_pred_classes[Y_pred > cutoff] = 1 
y_test_classes = np.zeros_like(y_test)
y_test_classes[y_test > cutoff] = 1
cm = confusion_matrix(y_test_classes, y_pred_classes)
print(cm)

fig, ax = plt.subplots(figsize=(10,5))
ax.matshow(cm)
plt.title("Confusion Matrix")
plt.ylabel("True")
plt.xlabel("Prediccion")
for (i,j), z in np.ndenumerate(cm):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
plt.show()


X = data.iloc[:, 0].values.reshape(-1, 1) # values converts it into a numpy array
Y = data.iloc[:, 1].values.reshape(-1, 1) # -1 means that calculate the dimension 
linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)
plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()
from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree = 2)
X_poly = pr.fit_transform(X)
pr.fit(X_poly, Y)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, Y)
plt.scatter(X, Y)
plt.scatter(X, lin_reg.predict(pr.fit_transform(X)))
plt.show()

