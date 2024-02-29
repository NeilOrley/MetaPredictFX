import sys
import MetaTrader5 as mt5
from MetaTrader5 import *
import matplotlib.pyplot as plt
import seaborn as sns  # Import seaborn for residual plot



def main(value1, value2, value3, value4, value5, value6, value7,value8):
    # Now you can use the variables value1, value2, value3, value4, value5, value6, value7 en tu programa
    print("Numero de cuenta mt5 en mt5: ", value1)
    print("Password en mt5: ", value2)
    print("Servidor en mt5: ", value3)
    print("Delay dias ticks: ", value4)
    print("Time frame: ", value5)
    print("Epochs: ", value6)
    print("Symbol: ", value7)
    print("Path a mt5: ", value8)
    

def execute_program_without(value1, value2, value3, value4, value5, value6, value7, value8):
    main(value1, value2, value3, value4, value5, value6, value7, value8)

    import numpy as np
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Import r2_score
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.regularizers import l2
    from sklearn.model_selection import KFold

    import time

    import random

    # display data on the MetaTrader 5 package
    print("MetaTrader5 package author: ",mt5.__author__)
    print("MetaTrader5 package version: ",mt5.__version__)

    # establish connection to the MetaTrader 5 terminal
    # display data on MetaTrader 5 version
    print(mt5.version())
    # connect to the trade account without specifying a password and a server


    account=int(value1)
    clave_secreta=str(value2)
    server_account=str(value3)

    
    delay_1=int(value4)
    cerrar1=60
    Seleccion_tmf=str(value5)

    epoch =int(value6)
    selection=str(value7)
    k_reg=0.001#input("Please input the kernel regulizer you whant to have (for example 0.01): ")
    i_mae=0.0001#input("Please input the MAE from whom you whant to filter: ")
    i_mse=0.001#input("Please input the MSE from whom you whant to filter: ")
    i_r2=0.75#input("Please input the r2 from whom you whant to filter (for example 0.8): ")

    # You will need to update the values for path, login, pass, and server according to your specific case.
    creds = {
        "path": str(value8),
        "login": account,
        "pass": clave_secreta,
        "server": server_account,
        "timeout": 60000,
        "portable": False
    }
    
    # Lanzamos la plataforma MT5 y nos conectamos al servidor con nuestro usuario y contraseña
    if mt5.initialize(path=creds['path'],
                    login=creds['login'],
                    password=creds['pass'],
                    server=creds['server'],
                    timeout=creds['timeout'],
                    portable=creds['portable']):
        
            print("Plataform MT5 launched correctly")
    else:
        print(f"There has been a problem with initialization: {mt5.last_error()}")

    if not mt5.initialize():
        print("initialize() failed, error code =",mt5.last_error())
        quit()

    import pandas as pd
    from datetime import datetime, timedelta
    #############################################################
    # Obtains actual time and date
    now_inicio_script = time.time()
    ###########################################################
    symbols=selection

    if Seleccion_tmf== "1":
        tmf = mt5.TIMEFRAME_H1
        seconds = 60*60
        delay=delay_1
        cerrar=cerrar1*60
        temporalidad="1 hora"
    if Seleccion_tmf== "2":
        tmf = mt5.TIMEFRAME_H2
        seconds = 60*60*2
        delay=delay_1
        cerrar=cerrar1*60*2
        temporalidad="2 horas"
    if Seleccion_tmf== "3":
        tmf = mt5.TIMEFRAME_H3
        seconds = 60*60*3
        delay=delay_1
        cerrar=cerrar1*60*3
        temporalidad="3 horas"
    if Seleccion_tmf== "4":
        tmf = mt5.TIMEFRAME_H4
        seconds = 60*60*4
        delay=delay_1
        cerrar=cerrar1*60*4
        temporalidad="4 horas"
    if Seleccion_tmf== "1d":
        tmf = mt5.TIMEFRAME_D1
        seconds = 60*60*24
        delay=delay_1
        cerrar=cerrar1*60*24
        temporalidad="1 dia"
    if Seleccion_tmf== "3d":
        tmf = mt5.TIMEFRAME_D1
        seconds = 60*60*24*3
        delay=delay_1
        cerrar=cerrar1*60*24*3
        temporalidad="3 dias"
    if Seleccion_tmf== "4d":
        tmf = mt5.TIMEFRAME_D1
        seconds = 60*60*24*4
        delay=delay_1
        cerrar=cerrar1*60*24*4
        temporalidad="4 dias"
    if Seleccion_tmf== "1w":
        tmf = mt5.TIMEFRAME_W1
        seconds = 60*60*24*7
        delay=delay_1
        cerrar=cerrar1*60*24*7
        temporalidad="1 semana"
    print(temporalidad)

    print(symbols)
    # Initialize an empty list to store results
    df = pd.DataFrame()

    df = df.drop(index=df.index)

    df2 = pd.DataFrame()
    # Empty DataFrame
    df2 = df2.drop(index=df2.index)

    # Create an empty dictionary to store DataFrames
    dfs = pd.DataFrame()
    # Empty the DataFrame
    dfs = dfs.drop(index=dfs.index)

    simbolito=symbols
    print("Symbol: ", simbolito)

    symbol = simbolito
    timeframe = temporalidad

    # import the 'pandas' module for displaying data obtained in the tabular form
    import pandas as pd
    pd.set_option('display.max_columns', 500) # number of columns to be displayed
    pd.set_option('display.width', 1500)      # max table width to display
    # import pytz module for working with time zone
    import pytz

    # stablish connection to MetaTrader 5 terminal
    if not mt5.initialize():
        print("initialize() failed, error code =",mt5.last_error())
        quit()

    # set time zone to UTC
    timezone = pytz.timezone("Etc/UTC")

    # Obtener la fecha y hora actual
    now=None
    now = datetime.now()

    # Print actual date and time in legible format
    print("Fecha y hora actual:", now)


    formatted_now=None
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Fecha y hora formateadas:", formatted_now)

    # Minus n days
    date_n_days_ago =None
    date_n_days_ago = now - timedelta(days=int(delay))

    # Print date from n days ago
    print("Fecha y hora hace n días:", date_n_days_ago)

    # También puedes formatear la salida según tus preferencias
    formated_date_n_days_ago = date_n_days_ago.strftime("%Y,%m,%d")
    formated_date_n_days_ago_y = date_n_days_ago.strftime("%Y")
    formated_date_n_days_ago_m = date_n_days_ago.strftime("%m")
    formated_date_n_days_ago_d = date_n_days_ago.strftime("%d")
    print("Fecha y hora formateadas hace n días:", formated_date_n_days_ago)

    # create 'datetime' object in UTC time zone to avoid the implementation of a local time zone offset
    utc_from = datetime(int(formated_date_n_days_ago_y),int(formated_date_n_days_ago_m),int(formated_date_n_days_ago_d), tzinfo=timezone)

    rates=pd.DataFrame()
    # Empty DataFrame
    rates = rates.drop(index=rates.index)
    rates = mt5.copy_ticks_from(symbol, utc_from, 1000000000, mt5.COPY_TICKS_ALL)
    print(rates)

    # create DataFrame out of the obtained data
    rates_frame=pd.DataFrame()
    # Vaciar el DataFrame
    rates_frame = rates_frame.drop(index=rates_frame.index)
    rates_frame = pd.DataFrame(rates)
    print(rates_frame)

    # convert time in seconds into the datetime format
    rates_frame['time']=pd.to_datetime(rates_frame['time'], unit='s')
    rates_frame['close']=(rates_frame['ask']+rates_frame['bid'])/2
    #rates_frame_['time']=pd.to_datetime(rates_frame_['time'], unit='s')
    # display data
    print("\nDisplay dataframe with data")
    print(rates_frame) 

    df=rates_frame
    # Create a target variable (e.g., predict the next n closing prices)
    ############################################################################################
    df['target'] = df['close']
    df['time'] =rates_frame['time']
    df['time_target'] = df['time'].sub(pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    df['time_target_seconds'] = df['time'].sub(pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    print("tabla df añadiendo",df)
    time_ticks=None
    time_ticks=int(((df['time_target'].iloc[-1]-df['time_target'].iloc[0])))/int(len(df))
    time_ticks=(round(time_ticks,2))

    print("time_1 de ticks",time_ticks)

    print("el time_1 que dura la muestra en seconds ",time_ticks)

    numero_de_filas= seconds
    filas_vacias = pd.DataFrame(np.nan, index=range(numero_de_filas), columns=df.columns)
    df = df._append(filas_vacias, ignore_index=True)
    df['target'] = df['close'].shift(-seconds)
    print("df modificada",df)


    df2=df[['close','target']]#,'time_target','time_target_seconds']]
    print("df",df)
    print("df2",df2)

    # Drop NaN values
    df=df.dropna()
    df2 = df2.dropna()
    print("df con dropna",df)
    print("df2 con dropna",df2)

    # Split the data into features (X) and target variable (y)
    X=[]
    y=[]
    X = df2[['close']]
    y = df2['target']

    # Split the data into training and testing sets
    X_train=[]
    X_test=[]
    y_train=[]
    y_test=[]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Standardize the features
    X_train_scaled=[]
    X_test_scaled=[]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build a neural network model
    model=None
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(k_reg)))
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(k_reg)))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(k_reg)))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(k_reg)))
    model.add(Dense(1, activation='linear'))

    # Compile the model[]
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=int(epoch), batch_size=256, validation_split=0.2, verbose=1)

    # Use the model to predict the next 4 instances
    X_predict=[]
    X_predict_scaled=[]

    predictions = pd.DataFrame()
    predictions=[]
    # Vaciar el DataFrame
    #predictions = predictions.drop(index=predictions.index)
    X_predict = df2.tail(seconds)[['close']]
    X_predict_scaled = scaler.transform(X_predict)
    predictions = model.predict(X_predict_scaled)

    # Print actual and predicted values for the next 4 instances
    print("Actual Value for the Last Instances:")
    print(df.tail(1)['close'].values)

    print("\nPredicted Value for the Next Instances:")
    print(predictions[:, 0])
    predictions=pd.DataFrame(predictions)

    ######################################################################
    import matplotlib.pyplot as plt
    import seaborn as sns  # Import seaborn for residual plot
    ######################################################3
    first_prediction=pd.DataFrame()

    first_prediction = first_prediction.drop(index=first_prediction.index)
    last_value=pd.DataFrame()

    last_value = last_value.drop(index=last_value.index)

    dff=pd.DataFrame()

    dff = dff.drop(index=dff.index)

    predictions01=pd.DataFrame()

    predictions01 = predictions01.drop(index=predictions01.index)

    first_prediction=pd.DataFrame(first_prediction)
    last_value=pd.DataFrame(last_value)
    predictions01=pd.DataFrame({
    'new_column': predictions.iloc[:,0]
    })
    print("predictions01",predictions01)
    last_value=[]
    cell_value=[]
    scalar=None
    first_prediction01=None
    last_value= df.tail(1)['close'].values
    last_value=pd.DataFrame(last_value)
    cell_value = last_value.at[0, 0]
    scalar = float(cell_value)
    first_prediction=predictions01.loc[0,'new_column']
    first_prediction01 = float(first_prediction)

    dff5=pd.DataFrame()
    print("#######################")
    print("First prediction value",first_prediction01)
    if first_prediction01 >= scalar:
        
        
        dff5=predictions01.loc[:,'new_column'] -(first_prediction01-scalar)
        
    if first_prediction01 < scalar:
        
        
        dff5=predictions01.loc[:,'new_column'] +(scalar-first_prediction01)
        
    print("Predictions adapted to last real value")
    dff5=pd.DataFrame(dff5)

    print("dff5",dff5)
    now_str = str(now).replace(" ", "_").replace(":", "-").replace(".", "-")

    #######################################################

    """# Calculate and print mean squared error
    mse = i_mse#mean_squared_error(y_test, model.predict(X_test_scaled))
    print(f"\nMean Squared Error: {mse}")

    # Calculate and print mean absolute error
    mae = i_mae#mean_absolute_error(y_test, model.predict(X_test_scaled))
    print(f"\nMean Absolute Error: {mae}")

    ##############################################################
    # Calculate and print R2 Score
    r2 = r2_score(y_test, model.predict(X_test_scaled))
    print(f"\nR2 Score: {r2}")"""


    print("Symbol is: ",symbol)
    print("Last close price we have is: ",df.tail(1)['target'].values)
    print("Last time close price we have is: ",df.tail(1)['time'].values)    

    difference=None
    predicted1=df.tail(1)['target'].values + (predictions.iloc[0,0]-df.tail(1)['target'].values)
    difference=(predictions.iloc[0,0]-df.tail(1)['target'].values)
    if difference>0:
        Higher_Lower1="Higher"
    else:
        Higher_Lower1="Lower"
    percentage11=None    
    percentage11=(100*predicted1/df.tail(1)['target'].values)-100
    print("Prediction for the next time period is: ",predicted1," and it will gow: ",Higher_Lower1," in a % : ",percentage11)
    results_df=pd.DataFrame()
    # Vaciar el DataFrame
    results_df = results_df.drop(index=results_df.index)
    results_df = pd.DataFrame(columns=['Indice','Last Close','Last time','Symbol', 'MSE', 'MAE', 'R2 Score', 'Prediction1', 'High or Low1', "Percent change1"])#,'Prediction2', 'High or Low2', "Percent change2", 'Prediction3', 'High or Low3', "Percent change3",'Prediction4', 'High or Low4', "Percent change4"])

    # Append results to the DataFrame
    results_df=pd.DataFrame(results_df)

    # Obtener la fecha y hora actual
    ahora = datetime.now()

    # Imprimir la fecha y hora actual en un formato legible
    print("Actual hour and date time: ", ahora)

    # trabajando sobre predicions 
    max_pred=None
    min_pred=None
    close_pred=None
    difference_close_pred_value=None
    difference_close_pred_value_percent=None
    max_pred=predictions.max()
    min_pred=predictions.min()
    close_pred=dff5.iloc[-1,0]
    difference_close_pred_value_percent= (close_pred*100/df.tail(1)['target'].values)-100
    difference_close_pred_value =close_pred - df.tail(1)['target'].values

    dfs = pd.DataFrame({
        #'Indice':ii,
        'Inicit script':now,
        'End script':ahora,
        'Last Close':df.tail(1)['target'].values,
        'First Time':df.head(1)['time'].values,
        'Last Time':df.tail(1)['time'].values,
        'Symbol': simbolito,
        'MSE': "Not chosen",
        'MAE': "Not chosen",
        'R2 Score': "Not chosen",
        'Prediction1':predicted1,
        'High or Low1':Higher_Lower1, 
        "Percent change1":percentage11,
        'max of predicitons':max_pred,
        'min of predictions':min_pred,
        'close of predictions':str(close_pred),
        'diference between close and last prediction': str(difference_close_pred_value),
        'diference between close and last prediction in percetage': str(difference_close_pred_value_percent)
        },)


    print(dfs)
    #############################################################################
    time_diference=None
    time_diference=ahora-now# ahora es el time_1 cuando ha terminado el script y now es fecha inicio script, dif time_1 es lo que ha tardado el script

    # Imprimir la difference en días, horas, minutos y seconds
    print(f"difference total time_1 tardado: {time_diference}")
    print(f"Días: {time_diference.days}")
    print(f"Horas: {time_diference.seconds // 3600}")
    print(f"Minutos: {(time_diference.seconds // 60) % 60}")
    print(f"seconds: {time_diference.seconds % 60}")

    from datetime import datetime, timedelta
    new_date=None
    new_date=now+ timedelta(seconds=cerrar)

    # Imprimir la nueva fecha
    print(f"Fecha inicial: {now}")
    print(f"seconds a agregar: {cerrar} seconds")
    print(f"Nueva fecha: {new_date}")

    dif_new_date=None
    dif_new_date = new_date - now

    # Imprimir la difference en días, horas, minutos y seconds
    print(f"difference total hasta final grafica: {dif_new_date}")
    print(f"Días: {dif_new_date.days}")
    print(f"Horas: {dif_new_date.seconds // 3600}")
    print(f"Minutos: {(dif_new_date.seconds // 60) % 60}")
    print(f"seconds: {dif_new_date.seconds % 60}")

    total_t_dif=None
    total_t_dif=(time_diference.days*24*60) +(time_diference.seconds //60) # 
    dif_new_date2=None
    dif_new_date2=(dif_new_date.days*24*60) +(dif_new_date.seconds // 60) # 
    time_to_know=None
    time_to_know= dif_new_date2-total_t_dif # 
    len_predicted=None
    len_predicted= len(predictions01) # 

    print("len_predicted: ",len_predicted)

    entry_point=None
    entry_point_perc=None
    entry_point_perc=100*(dif_new_date2-total_t_dif)/dif_new_date2

    entry_point_perc=int(round(entry_point_perc,0))

    ##########################################################################################################
    # Calculates difference in time
    time_spent = time.time() - now_inicio_script

    print(f"The script spent {time_spent} seconds.")
    ########################################################################################################

    entry_pto=total_t_dif/time_ticks
    entry_pto=int(round(entry_pto,0))
    entry_value=None
    entry_value = int(round(time_spent ,0))
    print("entry_value",entry_value)
    entry_point= entry_value #total_t_dif/time_ticks  

    entry_point=int(round(entry_point,0))

    time_left=None
    time_left=dif_new_date2-total_t_dif
    MyParameter=None
    MyParameter=int(round(time_left,0))

    len_predicitons=int(len(dff5))

    print("table dff5: ",dff5)
    print("table dff5 only col 0: ",dff5.iloc[0:,0])
    print("tabla from entry point: ",dff5.iloc[entry_point:,0])

    value_25=None

    value_25=seconds/5

    value_25=int(round(value_25,0))


    now_str = str(now).replace(" ", "_").replace(":", "-").replace(".", "-")

    ################################################################################################

    import requests

    #############################################################################
    pred_starts=None
    pred_starts=dff5.iloc[entry_point,0]
    print("pred_starts ",pred_starts)
    final_pred=None
    final_pred=dff5.iloc[-1,0]
    print("final_pred ",final_pred)
    dif_preds=None
    dif_preds=final_pred-pred_starts
    print("final difference predicciones",dif_preds)
    #######################################################################################

    def get_info(symbol):
            '''https://www.mql5.com/en/docs/integration/python_metatrader5/mt5symbolinfo_py
            '''
            # get symbol properties
            info=mt5.symbol_info(symbol)
            return info

    def open_trade_sell2(action, symbol, lot,random_integer, tp, sl, deviation):
            '''https://www.mql5.com/en/docs/integration/python_metatrader5/mt5ordersend_py
            '''
            # prepare the buy request structure
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
                "sl":sl,
                "tp":tp,
                "deviation": deviation,
                "magic": random_integer,
                "comment": "python open",
                "type_time": mt5.ORDER_TIME_GTC, # good till cancelled
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            # send a trading request
            result = mt5.order_send(buy_request)        
            return result, buy_request

    def open_trade_buy2(action, symbol, lot,random_integer, tp, sl, deviation):
            '''https://www.mql5.com/en/docs/integration/python_metatrader5/mt5ordersend_py
            '''
            # prepare the buy request structure
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
                "type_time": mt5.ORDER_TIME_GTC, # good till cancelled
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            # send a trading request
            result = mt5.order_send(buy_request)        
            return result, buy_request
    ######################################################################################
    def telegram_bot_sendtext2(bot_message):
        
        bot_token = 'xxx'
        bot_chatID = 'xxx'
        #bot_message_escaped = quote(bot_message)
        #send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message
        send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&text=' + bot_message
        #send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message_escaped

        response = requests.get(send_text)

        return response.json()
    ####################################################################################
    def close_position(position,symbol):
        """This function closes the position it receives as an argument."""
    
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'position': position.ticket,
            'magic': position.magic,
            'symbol': symbol,
            'volume': position.volume,
            'deviation': 50,
            'type': mt5.ORDER_TYPE_BUY if position.type == 1 else mt5.ORDER_TYPE_SELL,
            'type_filling': mt5.ORDER_FILLING_FOK,
            'type_time': mt5.ORDER_TIME_GTC,
            'comment': "mi primera orden desde Python"
        }
        return mt5.order_send(request)

    # Now, we define a new function that serves to close ALL open positions.
    def close_all_position(symbol):
        """This function closes ALL open positions and handles potential errors."""
    
        positions = mt5.positions_get()
        for position in positions:
            if close_position(position,symbol).retcode == mt5.TRADE_RETCODE_DONE:
                print(f"Position {position.ticket} closed correctly.")
            else:
                print(f"An error occurred while closing the position.{position.ticket}: {mt5.last_error()}")

    #####################################################################################
    percent1=None
    percent1=(100*float(dff5.iloc[entry_point,0])/float(dff5.iloc[-1,0]))-100
    maximal=None
    minimal=None

    print("seconds",seconds)
    print("entry_point",entry_point)
    print("value_25",value_25)
    maximal=(dff5.iloc[entry_point:value_25,0].max())
    minimal=(dff5.iloc[entry_point:value_25,0].min())

    # obteins index of maximal and minima
    entry_point=int(entry_point)
    print("entry_point",entry_point)
    value_25=int(value_25)
    maximal_index=None
    manimal_index=None
    initial_index=None
    final_index=None
    time_1=None
    maximal_index = dff5.iloc[entry_point:value_25, 0].idxmax()
    manimal_index = dff5.iloc[entry_point:value_25, 0].idxmin()
    initial_index = entry_point#dff5[dff5.iloc[:, 0] ==dff5.iloc[entry_point,0] ].index[0]
    final_index = value_25#dff5[dff5.iloc[:, 0] == dff5.iloc[value_25,0]].index[0]
    time_1 = MyParameter
    print("minimal",minimal)
    print("maximal",maximal)
    print("initial_index: ",initial_index)
    print("manimal_index: ",manimal_index)
    print("maximal_index: ",maximal_index)
    print("final_index: ",final_index)
    print("time_1 (min): ", time_1)
    dif_indexes_initial_final=None
    time_scale=None
    dif_indexes_initial_final= int(final_index) - int(initial_index)
    time_scale=1#(dif_indexes_initial_final/time_1)/60
    print("escala time_1 (m): ",time_scale)
    dif_min_max=None
    dif_min_max=maximal-minimal
    before=None
    percentage_min_max_var=None
    time_t1=None
    time_t2=None
    dif_t1_t2=None

    entry_point=int(entry_point)
    value_25=int(value_25)
    maximal_index=int(maximal_index)
    manimal_index=int(manimal_index)
    formatted_now2 = now.strftime("%Y_%m_%d_%H_%M_%S")
    print("Fecha y hora formateadas:", formatted_now)
    #################################################################################
    # Define el punto de entrada y otros puntos de división en el eje x
    center = len_predicitons/2  # Puedes ajustar este the_value según tus necesidades
    pt_division1 = len_predicitons/4
    pt_division2 = len_predicitons/2 + len_predicitons/4
    pt_division3 = len_predicitons/3
    pt_division5 = len_predicitons/5
    pt_division6 = len_predicitons/6
    pt_division10 = len_predicitons/10
    pt_division20 = len_predicitons/20
    pt_division14 = len_predicitons/14

    plt.axvline(x=pt_division2, color='gray', linestyle='--', label='75 %')
    plt.axvline(x=center, color='grey', linestyle='--', label='50 %')
    plt.axvline(x=pt_division3, color='blue', linestyle='--', label='33 %')
    plt.axvline(x=pt_division1, color='gray', linestyle='--', label='25 %')
    plt.axvline(x=pt_division6, color='blue', linestyle='--', label='16 %')
    plt.axvline(x=pt_division10, color='yellow', linestyle='--', label='10 %')
    plt.axvline(x=pt_division14, color='blue', linestyle='--', label='7 %')
    plt.axvline(x=pt_division20, color='yellow', linestyle='--', label='5 %')
    plt.axvline(x=entry_point, color='orange', linestyle='--', label='entrada')
    plt.axvline(x=value_25, color='orange', linestyle='--', label='salida 20%')
    plt.axvline(x=maximal_index, color='red', linestyle='--', label='maximal') ##### ni idea de porqué no pinta correctamente la linea
    plt.axvline(x=manimal_index, color='red', linestyle='--', label='minimal')# lo mismo aquí

    plt.plot(dff5.iloc[:, 0], linestyle='-', label='Predicted')

    plt.xlabel('Instances')
    plt.ylabel('Prediction Price')
    plt.legend()
    plt.title(f'Predicted {symbol} y quedan en minutos: ' + str(MyParameter))
    plt.savefig('Predicted_for_'+str(symbol)+'_quedan_'+str(MyParameter)+'_minutos_desde_'+str(formatted_now2)+'_.png')
    time.sleep(2)
    print("###############################")
    print("MSE value is: ", "not chosen")
    print("MAE value is: ", "not chosen")
    print("R2 score is: ","not chosen")
    print("###############################")
    #######################################################################################################################################################################################################
    ###########################################################################################################################
if __name__ == "__main__":
    value1, value2, value3, value4, value5, value6, value7 = sys.argv[1:]
    main(value1, value2, value3, value4, value5, value6, value7)
