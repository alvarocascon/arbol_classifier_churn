#modelo de regresion logistica para predecir churn en una telco

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



#Limpieza del DF
#carga
df = pd.read_csv('telecom_churn copia.txt')
#cambiar nombre
df.rename(columns={"Churn_num": "Churn"}, inplace=True)
#se transforma a numerica; 1,0 en vez de yes, no
df['Churn'] = df['Churn'].astype('int64')
#dumificamos y creamos un df con las variables voice y ip (voice mail plan/international calls.
vmp = pd.get_dummies(df['Voice mail plan'],drop_first=True,prefix="voice")
ip = pd.get_dummies(df['International plan'],drop_first=True,prefix="ip")
#eliminamos las columnas que había antes
df.drop(['Voice mail plan','International plan'],axis=1,inplace=True)
#unimos al df los dfs que hemos creado con las variables categóricas
df = pd.concat([df,vmp,ip],axis=1)
df.head()
df.drop('State',axis=1,inplace=True)
print(df.head())
columns_names = df.columns.values
#print (columns_names)


#MATRIZ DE ENTRENAMIENTO
# Inicializamos un árbol. Por ahora está vacío.
# Solo definimos cómo queremos que sea en cuanto a su estructura y condiciones de entrenamiento
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')
modelo = tree.DecisionTreeClassifier(max_depth=3)

# Entrenamos el árbol inicializado a partir de los datos que le pasemos con .fit()
cols = ['Account length','Area code','Number vmail messages','Total day minutes',
 'Total day calls', 'Total day charge', 'Total eve minutes',
 'Total eve calls' ,'Total eve charge' ,'Total night minutes',
 'Total night calls' ,'Total night charge', 'Total intl minutes',
 'Total intl calls', 'Total intl charge', 'Customer service calls', 'voice_Yes', 'ip_Yes']

X = df[cols]
y = df['Churn']
print ("X Shape",X.shape)
print ("y Shape",y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.25,
                                                    random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# Entreno el árbol con el set de entrenamiento
modelo = modelo.fit(X=X_train, y=y_train)
# Uso el árbol para predecir sobre el dataset de entrenamiento y de prueba
y_pred_train = modelo.predict(X_train)
y_pred_test = modelo.predict(X_test)

def entrenar_modelo_y_predecir_classificacion(modelo):
    # Entreno el árbol con el set de entrenamiento
  modelo = modelo.fit(X=X_train, y=y_train)
  # Uso el árbol para predecir sobre el dataset de entrenamiento
  y_pred_train = modelo.predict(X_train)
  # Uso el árbol para predecir sobre el dataset de test
  y_pred_test = modelo.predict(X_test)
  # Cómo de buena es la predicción?
  ac_train = round(accuracy_score(y_train, y_pred_train), 4)
  print('Precisión en set de entrenamiento :', ac_train)
  ac_test = round(accuracy_score(y_test, y_pred_test), 4)
  print('Precisión en set de test :', ac_test)
  print('Degradación: ', round((ac_train-ac_test)/ac_train*100,2), '%')

  # Inicializo un árbol con 10 de profundidad
  modelo = tree.DecisionTreeClassifier(max_depth=10)
  # Entrenamos y predecimos con dicho modelo
  entrenar_modelo_y_predecir_classificacion(modelo)

  # Inicializo un árbol con 15 de profundidad
  modelo = tree.DecisionTreeClassifier(max_depth=15)
  # Entrenamos y predecimos con dicho modelo
  entrenar_modelo_y_predecir_classificacion(modelo)


# O con la librería graphviz
import graphviz
# Export_graphviz
dot_data = tree.export_graphviz(modelo,
                                out_file=None,
                                feature_names=cols)
graph = graphviz.Source(dot_data)
print(graph)

# Cómo de buenas son las predicciones?
from sklearn.metrics import accuracy_score

ac_train = accuracy_score(y_train, y_pred_train)
ac_test = accuracy_score(y_test, y_pred_test)

print('Precisión en set de entrenamiento :', ac_train)
print('Precisión en set de test :', ac_test)
print('Degradación: ', round((ac_train-ac_test)/ac_train*100,2), '%')

#Datos Nuevos

