"""
Algoritmo Agrupamiento Jerárquico
"""
import pandas as pd
import matplotlib.pyplot as plt
#### CARGAR LOS DATOS ####
data = pd.read_csv('datasetVIH.csv')
### ANALIZAR LOS DATOS ###
#Conocer la forma de los datos
print(data.shape)
#Conocer los datos nulos
print(data.isnull().sum())
#Conocer el formato de los datos
data.dtypes
#Se selecionan unos datos al azar para posteriormente verificar el clúster 
#al que pertenecen
indices = [101,102,110]
muestras = pd.DataFrame(data.loc[indices], 
                       columns = data.keys()).reset_index(drop = True)
### PROCESAMIENTO DE LOS DATOS ###
#Eliminamos las columnas de región y canal 
data = data.drop(['molecularweight'], axis = 1)
muestras = muestras.drop(['molecularweight'], axis = 1)
#Se realiza el escalamiento de los datos
from sklearn import preprocessing
data_escalada = preprocessing.Normalizer().fit_transform(data)
muestras_escalada = preprocessing.Normalizer().fit_transform(muestras)
### ANÁLISIS DE MACHINE LEARNING ###
#Se determina las variables a evaluar
X = data_escalada
#Se gráfica el dendrograma para obtener el número de clúster
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Dendrogramas") 
dendrograma = shc.dendrogram(shc.linkage(X, method = 'ward'))
#Obtenido el número de clúster se procede a definir los clústeres 
from sklearn.cluster import AgglomerativeClustering
#Se define el algoritmo junto con el valor de K
algoritmo = AgglomerativeClustering(n_clusters = 2, 
                                    affinity='euclidean', linkage='ward')  
#Se entrena el algoritmo
algoritmo.fit(X)
pred1 = algoritmo.fit_predict(X)
#Utilicemos los datos de muestras y verifiquemos en que cluster se encuentran
muestra_prediccion = algoritmo.fit_predict(muestras_escalada)
for i, pred in enumerate(muestra_prediccion):
    print( "Muestra", i, "se encuentra en el clúster:", pred)
plt.show()
