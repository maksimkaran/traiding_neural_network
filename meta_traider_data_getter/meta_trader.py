import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import numpy as np
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import glob
from finta import TA
import math
from sklearn.preprocessing import StandardScaler
import os
def get_candle_data():
    filepaths = [file for file in glob.glob(f'D:/bruh/trading_deep_learning/stock_data/*.csv')]
    for p in tqdm(filepaths):
        try:
            candle_data = pd.read_csv(p)
        except Exception as e:
            print(f'Error with filepath {p}', repr(e))
        
        #candle_data = candle_data.drop(columns = 'datetime',axis = 1)

        candle_data['RSI'] = TA.RSI(candle_data,14)
        candle_data['ATR'] = TA.ATR(candle_data,14)

        candle_data['upper_gap'] = 1.2 * candle_data.ATR 
        candle_data['lower_gap'] = 1.6 * candle_data.ATR
        bbands = TA.BBANDS(candle_data,30,std_multiplier=2)
        candle_data = candle_data.join(bbands)
        bb_bandwith =(candle_data.BB_UPPER-candle_data.BB_LOWER)/candle_data.BB_MIDDLE
        candle_data['Bandwith'] = bb_bandwith
        candle_data =candle_data.dropna()  
        candle_data['buy'] = 0
        candle_data = candle_data.astype({'buy':int})
        symbol = p.split('\\')[-1].split('.')[0]
        candle_data.to_csv(f'D:/bruh/trading_deep_learning/stock_data/{symbol}.csv')

def get_buy_signal_and_standardise():
    filepaths = [file for file in glob.glob(f'D:/bruh/trading_deep_learning/stock_data/*.csv')]
    i = 0
 
    global X
    global Y
    prices = pd.DataFrame()
    for p in tqdm(filepaths):
        remover = False
        X = pd.DataFrame()
        Y = pd.DataFrame()
        try:
            candle_data = pd.read_csv(p)
        except Exception as e:
            print(f'Error with filepath {p}', repr(e))
        if candle_data.empty == True:
            
            continue
        long_entry_condition = pd.Series([])
        for i in candle_data.index:
            try:
                long_entry_condition = candle_data.shift(-5).close+(candle_data.open*0.02) > candle_data.open
            except:
                print("final trades")

        candle_data.loc[long_entry_condition,'buy'] = 1
        bought = candle_data[candle_data.buy == 1]
        not_bought = candle_data[candle_data.buy == 0]
        if (len(not_bought) == 0) or (len(bought) == 0):
                continue
        if len(bought)>len(not_bought):
            bought = bought.sample(n=len(not_bought))
        else:
            not_bought = not_bought.sample(n=len(bought))
        try:  
            bought = bought.sample(n=(int(bars_to_fetch/2)))
            not_bought = not_bought.sample(n=int((bars_to_fetch/2)))
        except:
            remover = True
            
            
        candle_data = pd.concat([bought,not_bought], axis = 0)
   
        Y = pd.concat([candle_data,Y])
        #candle_data.drop(columns = 'buy',axis = 1, inplace=True)
        candle_data.drop(columns = 'datetime',axis = 1, inplace=True)

        X = pd.concat([candle_data,X])
        Y  = Y['buy']
        column_to_exclude = 'buy'

        # Select columns to scale (all columns except 'C')
        columns_to_scale = X.columns[X.columns != column_to_exclude]
        scaler = StandardScaler()
        scaler.fit(X[columns_to_scale])
        standardised_data = scaler.transform(X[columns_to_scale])
        X[columns_to_scale] = standardised_data 
        X.drop(columns = 'Unnamed: 0',axis = 1, inplace=True)
        X.to_csv(p)
        if remover == True:
            print(f"File {p} has been removed.")
            os.remove(p)



TIMEFRAMES = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1']
TIMEFRAME_DICT = {
    'M1': mt5.TIMEFRAME_M1,
    'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
    'M30': mt5.TIMEFRAME_M30,
    'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4,
    'D1': mt5.TIMEFRAME_D1,
    'W1': mt5.TIMEFRAME_W1,
    'MN1': mt5.TIMEFRAME_MN1,
}
mt5.initialize()
login = 5031751061
password = 'S_WsFzH1'
server = 'MetaQuotes-Demo'
mt5.login(login,password,server)

'''def get_symbol_names():
    # connect to MetaTrader5 platform

    # get symbols
    symbols = mt5.symbols_get()
    print(symbols)
    symbols_df = pd.DataFrame(symbols, columns=symbols[0]._asdict().keys())

    symbol_names = symbols_df['name'].tolist()
    return symbol_names'''
tickers = pd.read_csv('tickers.csv')
symbols = tickers['ACT Symbol'].unique()
num = 0
bars_to_fetch = 40
bars_to_scan = 150
start_date = datetime(2024, 10, 1)
start_timestamp = int(start_date.timestamp())
if(input("da li preuzimas tikere? Y/N ") == 'Y'):
    for s in tqdm(symbols[0:506]):# C:\Users\maksim\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\bases\MetaQuotes-Demo\history
            candle_data = pd.DataFrame(mt5.copy_rates_from(s,mt5.TIMEFRAME_D1,start_timestamp,bars_to_scan))#datetime.now()))
            if len(candle_data.index) >=bars_to_scan-1:
                candle_data['datetime'] = pd.to_datetime(candle_data['time'], unit='s')
                num = num+1
                #candle_data.insert(column='datetime',value=date_time)
                candle_data = candle_data.set_index('datetime', inplace=False)

                candle_data.to_csv(f'D:/bruh/trading_deep_learning/stock_data/{s}.csv')
    get_candle_data()
get_buy_signal_and_standardise()