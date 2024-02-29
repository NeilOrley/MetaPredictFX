import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
import time
import random
import pytz
import requests


# Paramètres de configuration
symbol = "EURUSD"
seleccion_tmf = "4h"
delay = 900
cerrar = 60

account = 11111
password = "password"
server = "Server"
Lote = 0.1
epoch = 1
seconds = 60 * 60 * 4
selection = "EURUSD"
k_reg = 0.001
i_mae = 0.0001
i_mse = 0.001
i_r2 = 0.80

# Initialisation de MT5
def initialize_mt5():
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()
    print("MetaTrader5 package author: ", mt5.__author__)
    print("MetaTrader5 package version: ", mt5.__version__)
    print(mt5.version())

# Connexion à MT5 avec des crédits spécifiques
def connect_mt5_with_credentials():
    creds = {
        "path": "C:/Program Files/XXX MT5/terminal64.exe",
        "login": account,
        "password": password,
        "server": server,
        "timeout": 60000,
        "portable": False
    }
    if mt5.initialize(**creds):
        print("MT5 platform launched correctly")
    else:
        print(f"Initialization problem: {mt5.last_error()}")



def setup_and_fetch_data(symbol, seleccion_tmf, delay, cerrar):
    """
    Initialise MT5, sélectionne le timeframe, récupère et prétraite les données de marché.
    
    :param symbol: Symbole pour lequel récupérer les données
    :param seleccion_tmf: Sélection du timeframe (ex: "1", "2", "1d", "1w")
    :param delay: Délai en jours pour récupérer les données
    :param cerrar: Non utilisé dans cette fonction, mais conservé pour la cohérence
    :return: DataFrame des données récupérées et prétraitées
    """
    now_inicio_script = time.time()
    print(f"Symbol: {symbol}")
    
    # Définition du timeframe basé sur la sélection
    timeframes = {
        "1h": mt5.TIMEFRAME_H1,
        "2h": mt5.TIMEFRAME_H2,
        "3h": mt5.TIMEFRAME_H3,
        "4h": mt5.TIMEFRAME_H4,
        "1d": mt5.TIMEFRAME_D1,
        "1w": mt5.TIMEFRAME_W1,
    }
    tmf = timeframes.get(seleccion_tmf, mt5.TIMEFRAME_D1)
    
    # Initialisation de MT5
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()
    
    # Définition du fuseau horaire sur UTC
    timezone = pytz.timezone("Etc/UTC")
    utc_from = datetime.now(tz=timezone) - timedelta(days=delay)
    
    # Récupération des données
    rates = mt5.copy_ticks_from(symbol, utc_from, 1000000000, mt5.COPY_TICKS_ALL)
    rates_frame = pd.DataFrame(rates)
    
    # Prétraitement des données
    rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
    rates_frame['close'] = (rates_frame['ask'] + rates_frame['bid']) / 2
    
    print("\nDisplay dataframe with data")
    print(rates_frame)
    
    return rates_frame


def prepare_data(df):
    """
    Prépare les données pour le modèle en créant une variable cible
    et en divisant les données en ensembles d'entraînement et de test.
    """
    df['target'] = df['close'].shift(-1)  # Prédire le prix de clôture suivant
    df = df.dropna()  # Supprimer les valeurs NaN résultant du décalage
    
    # Séparation des caractéristiques (X) et de la variable cible (y)
    X = df[['close']]
    y = df['target']

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Standardisation des caractéristiques
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def build_and_train_model(X_train, y_train):
    """
    Construit et entraîne le modèle de deep learning.
    """
    model = Sequential([
        Dense(128, activation='relu', input_dim=X_train.shape[1], kernel_regularizer=l2(k_reg)),
        Dense(256, activation='relu', kernel_regularizer=l2(k_reg)),
        Dense(128, activation='relu', kernel_regularizer=l2(k_reg)),
        Dense(64, activation='relu', kernel_regularizer=l2(k_reg)),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epoch, batch_size=256, validation_split=0.2, verbose=1)
    return model

def make_predictions(model, X_predict_scaled):
    """
    Utilise le modèle pour faire des prédictions.
    """
    predictions = model.predict(X_predict_scaled)
    return predictions


def process_and_visualize_predictions(predictions, df):
    """
    Ajuste les prédictions par rapport à la dernière valeur réelle et visualise les résultats.
    
    :param predictions: Prédictions du modèle.
    :param df: DataFrame original contenant les données de marché.
    """
    # Préparation des DataFrames pour le traitement
    predictions01 = pd.DataFrame({'new_column': predictions[:, 0]})
    print("predictions01", predictions01)

    # Obtention de la dernière valeur réelle connue
    last_value = df['close'].iloc[-1]
    print("#######################")
    print("First prediction value", predictions01.iloc[0, 0])

    # Ajustement des prédictions par rapport à la dernière valeur réelle
    dff5 = predictions01.apply(lambda x: x + (last_value - x.iloc[0]) if x.iloc[0] < last_value else x - (x.iloc[0] - last_value))

    print("Predictions adapted to last real value")
    print(dff5)

    # Visualisation des prédictions ajustées
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=dff5, markers=True, dashes=False)
    plt.title('Predictions Adjusted to Last Real Value')
    plt.xlabel('Time Steps')
    plt.ylabel('Predicted Close Price')
    plt.show()

    return dff5


def evaluate_and_summarize_predictions(y_test, predictions, df, dff5):
    """
    Évalue les prédictions avec MSE, MAE, R2 et résume les résultats.

    :param y_test: Valeurs réelles pour tester.
    :param predictions: Prédictions du modèle.
    :param df: DataFrame original utilisé pour l'entraînement/le test.
    :param dff5: DataFrame des prédictions ajustées.
    """
    # Evaluation des prédictions
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\nMean Squared Error: {mse}")
    print(f"\nMean Absolute Error: {mae}")
    print(f"\nR2 Score: {r2}")

    # Analyse des prédictions par rapport à la dernière valeur connue
    last_close_price = df['close'].iloc[-1]
    predicted_next = predictions[0]
    difference = predicted_next - last_close_price
    direction = "Higher" if difference > 0 else "Lower"
    percent_change = (100 * predicted_next / last_close_price) - 100

    print(f"Symbol is: {selection}")
    print(f"Last close price we have is: {last_close_price}")
    print(f"Prediction for the next time period is: {predicted_next}, and it will go: {direction} in a % : {percent_change}")

    # Résumé des résultats
    max_pred = dff5['new_column'].max()
    min_pred = dff5['new_column'].min()
    close_pred = dff5['new_column'].iloc[-1]
    difference_close_pred_value = close_pred - last_close_price
    difference_close_pred_value_percent = (100 * close_pred / last_close_price) - 100

    print("Max of predictions:", max_pred)
    print("Min of predictions:", min_pred)
    print("Close of predictions:", close_pred)
    print("Difference between close and last prediction:", difference_close_pred_value)
    print("Difference between close and last prediction in percentage:", difference_close_pred_value_percent)



# Envoi de notifications via Telegram
def send_telegram_notification(message):
    bot_token = 'your_bot_token'
    bot_chatID = 'your_chat_id'
    send_text = f'https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={bot_chatID}&parse_mode=Markdown&text={message}'
    response = requests.get(send_text)
    return response.json()



def calculate_and_display_time_info(now_inicio_script, now, cerrar, predictions01, dff5):
    """
    Calcule et affiche les informations de temps, y compris la durée du script et les prédictions temporelles.

    :param now_inicio_script: Timestamp du début du script.
    :param now: Datetime du début du script.
    :param cerrar: Nombre de secondes pour calculer la nouvelle date à partir de 'now'.
    :param predictions01: DataFrame des premières prédictions ajustées.
    :param dff5: DataFrame des prédictions ajustées finales.
    """
    # Calcul et affichage des différences de temps
    ahora = datetime.now()
    time_difference = ahora - now
    print(f"Total script duration: {time_difference}")
    print(f"Days: {time_difference.days}, Hours: {time_difference.seconds // 3600}, Minutes: {(time_difference.seconds // 60) % 60}, Seconds: {time_difference.seconds % 60}")

    # Calcul de la nouvelle date après l'ajout de secondes
    new_date = now + timedelta(seconds=cerrar)
    diff_new_date = new_date - now
    print(f"Initial date: {now}, Seconds to add: {cerrar}, New date: {new_date}")
    print(f"Total difference until the end of graph: {diff_new_date}")

    # Calcul du temps total passé et affichage du tableau des prédictions ajustées
    total_time_diff = (time_difference.total_seconds() // 60)
    diff_new_date2 = (diff_new_date.total_seconds() // 60)
    time_to_know = diff_new_date2 - total_time_diff
    print(f"Length of adjusted predictions: {len(dff5)}")

    # Affichage du temps passé
    time_spent = time.time() - now_inicio_script
    print(f"The script spent {time_spent:.2f} seconds.")


# Début des nouvelles fonctions fournies
def get_info(symbol):
    """https://www.mql5.com/en/docs/integration/python_metatrader5/mt5symbolinfo_py"""
    info = mt5.symbol_info(symbol)
    return info

def open_trade_sell(action, symbol, lot, random_integer, tp, sl, deviation):
    """https://www.mql5.com/en/docs/integration/python_metatrader5/mt5ordersend_py"""
    symbol_info = get_info(symbol)
    if action == 'buy':
        trade_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).ask
    elif action =='sell':
        trade_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).bid
    point = mt5.symbol_info(symbol).point
    print("el precio mt5 es:", price)

    buy_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": trade_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": deviation,
        "magic": random_integer,
        "comment": "python open",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(buy_request)        
    return result, buy_request

def open_trade_buy(action, symbol, lot, random_integer, tp, sl, deviation):
    symbol_info = get_info(symbol)
    if action == 'buy':
        trade_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).ask
    elif action == 'sell':
        trade_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).bid
    buy_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": trade_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": deviation,
        "magic": random_integer,
        "comment": "python open",
        "type_time": mt5.ORDER_TIME_GTC,  # good till cancelled
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    # Envoi d'une requête de trading
    result = mt5.order_send(buy_request)
    return result, buy_request


def telegram_bot_sendtext2(bot_message):
    bot_token = 'your_bot_token'  # Remplacer par votre token de bot
    bot_chatID = 'your_chat_id'  # Remplacer par votre chat ID
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&text=' + bot_message
    response = requests.get(send_text)
    return response.json()


def close_position(position, symbol):
    request = {
        'action': mt5.TRADE_ACTION_DEAL,
        'position': position.ticket,
        'magic': position.magic,
        'symbol': symbol,
        'volume': position.volume,
        'deviation': 50,
        'type': mt5.ORDER_TYPE_BUY if position.type == mt5.ORDER_TYPE_SELL else mt5.ORDER_TYPE_SELL,
        'type_filling': mt5.ORDER_FILLING_FOK,
        'type_time': mt5.ORDER_TIME_GTC,
        'comment': "Close order from Python"
    }
    result = mt5.order_send(request)
    return result

def close_all_position(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if positions == None or len(positions) == 0:
        print("No positions to close for symbol", symbol)
        return
    for position in positions:
        result = close_position(position, symbol)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"Position {position.ticket} closed correctly.")
        else:
            print(f"Failed to close position {position.ticket}: {mt5.last_error()}")

# Fonction principale
def main():
    now_inicio_script = time.time()
    now = datetime.now()
    cerrar = 60 * 60 * 4  # Pour cet exemple, 4 heures

    initialize_mt5()
    connect_mt5_with_credentials()
    df = setup_and_fetch_data(symbol, seleccion_tmf, delay, cerrar)
    X_train_scaled, X_test_scaled, y_train, y_test = prepare_data(df)
    model = build_and_train_model(X_train_scaled, y_train)
    predictions = make_predictions(model, X_test_scaled)
    dff5 = process_and_visualize_predictions(predictions, df) #DataFrame des prédictions ajustées finales
    evaluate_and_summarize_predictions(y_test, predictions, df, dff5)
    calculate_and_display_time_info(now_inicio_script, now, cerrar, predictions, dff5)
    
   
if __name__ == "__main__":
    main()