import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("EjemploEstudiantes.csv", sep=";",decimal=',', index_col=0)
print(df)
#Tomamos los valores del csv, que son las materias
X = df.iloc[:,[0,1,2,3,4]].values
print(X)
#y = df.iloc[:,4].values
#Paso 1 Centrar y Reducir la tabla de datos de X
X_std = StandardScaler().fit_transform(X)
#Paso 2 Matriz de correlaciones
print('Covarianza Matrix: \n%s' %np.cov(X_std.T))

#Paso 3 Calcular los vectores y valores propios de la matrix
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)
#Paso 4 Ordenar de mayor a menor estos valores propios
print('Autovalores en orden descendiente:')
for i in eig_pairs:
    print(i[0])

#Paso 5 
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

with plt.style.context('seaborn-pastel'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(5), var_exp, alpha=0.5, align='center',
            label='Vindividual', color='g')
    plt.step(range(5), cum_var_exp, where='mid', linestyle='--', label='Vacumulada')
    plt.ylabel('RVarianza')
    plt.xlabel('PC')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
#Paso 6 Calcule matriz de componentes principales
matrix_w = np.hstack((eig_pairs[0][1].reshape(5,1),
                      eig_pairs[1][1].reshape(5,1)))
print('Components matrix:\n', matrix_w)
Y = X_std.dot(matrix_w)

   

