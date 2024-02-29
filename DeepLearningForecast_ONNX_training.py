import MetaTrader5 as mt5
from MetaTrader5 import *
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Importer r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import KFold
import time
import random
import pandas as pd
from datetime import datetime, timedelta
# Afficher les données sur le package MetaTrader 5

# Se connecter au compte de trading sans spécifier de mot de passe ni de serveur

account=xxx #input("Entrez votre numéro de compte : ")
clave_secreta="xxxxxxxxxxxxxxx" #input("Entrez votre mot de passe : ")
server_account="xxxxxxx" #input("Entrez le serveur pour le compte : ")

delay_1=900 #input("Veuillez entrer le délai en jours à partir duquel obtenir les ticks (mieux vaut plus de données que d'époques) : ")
cerrar1=60
Seleccion_tmf="1d" #input("Veuillez sélectionner l'intervalle de temps pour la prévision : 1 heure (appuyez sur : 1), 2 heures (2), 4 heures (4) ou 1 jour (1d), (3d), (4d), (1w) : ")

epoch =100 # input("Veuillez entrer le nombre d'époques à utiliser (par exemple 75,100 ou 150) : ")
selection="EURUSD" #input("Veuillez entrer le symbole tel qu'il apparaît chez votre courtier : ")
k_reg=0.001 #input("Veuillez entrer le régularisateur de noyau que vous souhaitez avoir (par exemple 0.01) : ")

k_reg=float(k_reg)

# Obtient l'heure et la date actuelles
now_inicio_script = time.time()

symbols=selection

if Seleccion_tmf == "1":
    tmf = mt5.TIMEFRAME_H1
    seconds = 60*60
    delay=delay_1
    cerrar=cerrar1*60
    temporalidad="1 heure"
if Seleccion_tmf == "2":
    tmf = mt5.TIMEFRAME_H2
    seconds = 60*60*2
    delay=delay_1
    cerrar=cerrar1*60*2
    temporalidad="2 heures"
if Seleccion_tmf == "3":
    tmf = mt5.TIMEFRAME_H3
    seconds = 60*60*3
    delay=delay_1
    cerrar=cerrar1*60*3
    temporalidad="3 heures"
if Seleccion_tmf == "4":
    tmf = mt5.TIMEFRAME_H4
    seconds = 60*60*4
    delay=delay_1
    cerrar=cerrar1*60*4
    temporalidad="4 heures"
if Seleccion_tmf == "1d":
    tmf = mt5.TIMEFRAME_D1
    seconds = 60*60*24
    delay=delay_1
    cerrar=cerrar1*60*24
    temporalidad="1 jour"
if Seleccion_tmf == "3d":
    tmf = mt5.TIMEFRAME_D1
    seconds = 60*60*24*3
    delay=delay_1
    cerrar=cerrar1*60*24*3
    temporalidad="3 jours"
if Seleccion_tmf == "4d":
    tmf = mt5.TIMEFRAME_D1
    seconds = 60*60*24*4
    delay=delay_1
    cerrar=cerrar1*60*24*4
    temporalidad="4 jours"
if Seleccion_tmf == "1w":
    tmf = mt5.TIMEFRAME_W1
    seconds = 60*60*24*7
    delay=delay_1
    cerrar=cerrar1*60*24*7
    temporalidad="1 semaine"
print(temporalidad)

print(symbols)
# Initialiser une liste vide pour stocker les résultats
df = pd.DataFrame()

df = df.drop(index=df.index)

df2 = pd.DataFrame()
# DataFrame vide
df2 = df2.drop(index=df2.index)

# Créer un dictionnaire vide pour stocker les DataFrames
dfs = pd.DataFrame()
# Vider le DataFrame
dfs = dfs.drop(index=dfs.index)

simbolito=symbols
print("Symbole : ", simbolito)

symbol = simbolito
timeframe = temporalidad

# Importer le module 'pandas' pour afficher les données obtenues sous forme tabulaire
import pandas as pd
pd.set_option('display.max_columns', 500) # Nombre de colonnes à afficher
pd.set_option('display.width', 1500)      # Largeur maximale du tableau à afficher
# Importer le module pytz pour travailler avec le fuseau horaire
import pytz

# Définir le fuseau horaire sur UTC
timezone = pytz.timezone("Etc/UTC")

# Obtenir la date et l'heure actuelles
now=None
now = datetime.now()

# Imprimer la date et l'heure actuelles dans un format lisible
print("Date et heure actuelles :", now)

formatted_now=None
formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
print("Date et heure formatées :", formatted_now)

# Moins n jours
date_n_days_ago =None
date_n_days_ago = now - timedelta(days=int(delay))

# Imprimer la date d'il y a n jours
print("Date et heure il y a n jours :", date_n_days_ago)

formated_n_time=date_n_days_ago.strftime("%Y-%m-%d %H:%M:%S")
print("Date et heure il y a n jours",formated_n_time)

# Vous pouvez également formater la sortie selon vos préférences
formated_date_n_days_ago = date_n_days_ago.strftime("%Y,%m,%d")
formated_date_n_days_ago_y = date_n_days_ago.strftime("%Y")
formated_date_n_days_ago_m = date_n_days_ago.strftime("%m")
formated_date_n_days_ago_d = date_n_days_ago.strftime("%d")
print("Date et heure formatées il y a n jours :", formated_date_n_days_ago)

# Créer un objet 'datetime' dans le fuseau horaire UTC pour éviter l'application d'un décalage de fuseau horaire local
utc_from = datetime(int(formated_date_n_days_ago_y),int(formated_date_n_days_ago_m),int(formated_date_n_days_ago_d), tzinfo=timezone)

# C'est le chemin complet vers le fichier csv des données
path_csv = 'xxxxxxxxxxxx/MQL5/Files/'

rates=pd.DataFrame()
# DataFrame vide
rates = rates.drop(index=rates.index)

import os

file_path = path_csv + 'ticks_data.csv'

if os.path.exists(file_path):
    rates = pd.read_csv(file_path, encoding='utf-16le')
else:
    print(f"Erreur : Fichier non trouvé - {file_path}")
    quit()
#rates = pd.read_csv(path_csv+str(symbol)+'_TickData.csv')

#rates = mt5.copy_ticks_from(symbol, utc_from, 1000000000, mt5.COPY_TICKS_ALL)
print(rates)

# Créer un DataFrame à partir des données obtenues
rates_frame=pd.DataFrame()
# Vider le DataFrame
rates_frame = rates_frame.drop(index=rates_frame.index)
rates_frame = pd.DataFrame(rates)
rates = rates.drop(index=rates.index)
print(rates_frame)
# Renommer les colonnes
rates_frame.rename(columns={0: 'Time', 1: 'bid', 2 : 'ask',3:'spread'}, inplace=True)

# Étape 2 : Convertir la colonne 'Time' au format datetime de pandas
rates_frame['Time'] = pd.to_datetime(rates_frame['Time'], format='%Y.%m.%d %H:%M', errors='coerce')
# Vérifier s'il y a des dates nulles après la conversion
if rates_frame['Time'].isnull().any():
    print("Il y a des dates invalides dans la colonne 'timestamp'.")

# En supposant que votre DataFrame rates_frame a une colonne 'timestamp'
# Convertir la colonne 'timestamp' au type datetime
#rates_frame['timestamp'] = pd.to_datetime(rates_frame['timestamp'])

# Sélectionner la plage de dates que vous souhaitez
fecha_inicio = formated_n_time #'2023-01-01'
fecha_fin =formatted_now #'2023-12-31'
print("date de début",fecha_fin)
print("date de fin",fecha_fin)

# Filtrer le DataFrame pour obtenir uniquement les lignes dans la plage de dates
rates_frame_filtrado = rates_frame[(rates_frame['Time'] >= fecha_inicio) & (rates_frame['Time'] <= fecha_fin)]

# Imprimer le DataFrame filtré
print(rates_frame_filtrado)

# Étape 4 : Trier le DataFrame par la colonne 'Time'
rates_frame_filtrado = rates_frame_filtrado.sort_values(by='Time')

# Afficher le DataFrame trié avec 'Time_seconds'
print(rates_frame_filtrado['Time'])

# Convertir le temps en secondes au format datetime

#rates_frame_filtrado['time']=pd.to_datetime(rates_frame_filtrado['Time'], unit='s')
rates_frame_filtrado['close']=(rates_frame_filtrado['Ask']+rates_frame_filtrado['Bid'])/2

# Afficher les données
print("\nAfficher le dataframe avec les données")
print(rates_frame_filtrado) 

# Créer une variable cible (par exemple, prédire les n prochains prix de clôture)
############################################################################################
rates_frame_filtrado['target'] = rates_frame_filtrado['close']
rates_frame_filtrado['time'] =rates_frame_filtrado['Time']
rates_frame_filtrado['time_target'] = rates_frame_filtrado['time'].sub(pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
rates_frame_filtrado['time_target_seconds'] = rates_frame_filtrado['time'].sub(pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
print("table rates_frame_filtrado ajoutant",rates_frame_filtrado)
time_ticks=None
time_ticks=int(((rates_frame_filtrado['time_target'].iloc[-1]-rates_frame_filtrado['time_target'].iloc[0])))/int(len(rates_frame_filtrado))
time_ticks=(round(time_ticks,2))

print("time_1 de ticks",time_ticks)

print("le time_1 mesure le temps d'obtention des données en secondes ",time_ticks)

number_of_rows= seconds
empty_rows = pd.DataFrame(np.nan, index=range(number_of_rows), columns=rates_frame_filtrado.columns)
rates_frame_filtrado = rates_frame_filtrado._append(empty_rows, ignore_index=True)
rates_frame_filtrado['target'] = rates_frame_filtrado['close'].shift(-seconds)
print("rates_frame_filtrado modifié",rates_frame_filtrado)

df2=rates_frame_filtrado[['close','target','time']]

print("rates_frame_filtrado",rates_frame_filtrado)
print("df2",df2)

# Supprimer les valeurs NaN
rates_frame_filtrado=rates_frame_filtrado.dropna()
rates_frame_filtrado = rates_frame_filtrado.drop(index=rates_frame_filtrado.index)
df2 = df2.dropna()
print("rates_frame_filtrado avec dropna",rates_frame_filtrado)
print("df2 avec dropna",df2)

# Diviser les données en caractéristiques (X) et variable cible (y)
X=[]
y=[]
X = df2[['close']]
y = df2['target']

# Diviser les données en ensembles d'entraînement et de test
X_train=[]
X_test=[]
y_train=[]
y_test=[]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Standardiser les caractéristiques
X_train_scaled=[]
X_test_scaled=[]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Construire un modèle de réseau de neurones
model=None
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(k_reg)))
model.add(Dense(256, activation='relu', kernel_regularizer=l2(k_reg)))
model.add(Dense(128, activation='relu', kernel_regularizer=l2(k_reg)))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(k_reg)))
model.add(Dense(1, activation='linear'))

# Compiler le modèle
model.compile(optimizer='adam', loss='mean_squared_error')

giggs=model.summary()
print(giggs)

time_calc_start = time.time()
# Entraîner le modèle
history=model.fit(X_train_scaled, y_train, epochs=int(epoch), batch_size=256, validation_split=0.2, verbose=1)
print(history)
fit_time_seconds = time.time() - time_calc_start
print("temps d'ajustement =",fit_time_seconds," secondes.")
# Utiliser le modèle pour prédire les 4 prochaines instances
X_predict=[]
X_predict_scaled=[]

predictions = pd.DataFrame()
predictions=[]
# Vider le DataFrame
#predictions = predictions.drop(index=predictions.index)
X_predict = df2.tail(seconds)[['close']]
X_predict_scaled = scaler.transform(X_predict)
predictions = model.predict(X_predict_scaled)

# Imprimer les valeurs réelles et prédites pour les 4 prochaines instances
print("Valeur réelle pour les dernières instances :")
print(df2.tail(1)['close'].values)

print("\nValeur prédite pour les prochaines instances :")
print(predictions[:, 0])
predictions=pd.DataFrame(predictions)

    ######################################################### ONNX
# Bibliothèques python
import MetaTrader5 as mt5
import tensorflow as tf
import numpy as np
import pandas as pd
import tf2onnx
from sklearn.model_selection import train_test_split
from sys import argv
import matplotlib.pyplot as plt 
import pytz

symbol=symbol
x_train=X_train
x_test=X_test
inp_model_name = "model."+str(symbol)+".onnx"
file_path="xxxxxxxxxxxxxxxxxxxx/MQL5/Files/"
data_path=argv[0]
last_index=data_path.rfind("\\")+1
data_path=data_path[0:last_index]
print("Chemin des données pour sauvegarder le modèle onnx",data_path)


# Et sauvegarder dans le dossier MQL5\Files pour utiliser comme fichier


print("Chemin du fichier pour sauvegarder le modèle onnx",file_path)

# Sauvegarder le modèle au format ONNX
output_path = data_path+inp_model_name
onnx_model = tf2onnx.convert.from_keras(model, output_path=output_path)
print(f"modèle sauvegardé dans {output_path}")


#####################################################################

# Calculer les métriques
from sklearn import metrics
from sklearn.metrics import r2_score

# Calculer et imprimer l'erreur quadratique moyenne
mse = mean_squared_error(y_test, model.predict(X_test_scaled))
print(f"\nErreur Quadratique Moyenne : {mse}")

# Calculer et imprimer l'erreur absolue moyenne
mae = mean_absolute_error(y_test, model.predict(X_test_scaled))
print(f"\nErreur Absolue Moyenne : {mae}")

# Calculer et imprimer le Score R2
r2 = r2_score(y_test, model.predict(X_test_scaled))
print(f"\nScore R2 : {r2}")

print("Terminé")
#mt5.shutdown
print("##################################################################")
