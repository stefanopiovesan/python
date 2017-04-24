# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 18:03:50 2017

@author: ProInspect
"""

import pandas as pd  # time series management
from pandas_datareader import data as web  # data retrieval
import seaborn as sns; sns.set()  # nicer plotting style

def get_data(symbols, start_date, end_date):
    dates=pd.date_range(start_date, end_date)
    df = pd.DataFrame(index=dates)
    
    for symbol in symbols:
        df_temp = web.DataReader(symbol, data_source='yahoo')
        df_temp.drop(df_temp.columns[range(5)], axis=1, inplace=True)
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df=df.join(df_temp)
    df=df.dropna()
    return df

def plot_bollinger_bands(df):
    rm = df.rolling(window=20,center=False).mean()
    rstd = df.rolling(window=20,center=False).std()
    upper_band = rm + rstd*2.0
    lower_band = rm - rstd*2.0
    # Plot raw SPY values, rolling mean and Bollinger Bands
    ax = df.plot(title="Bollinger Bands1", label='t')
    df.plot(label='Rolling mean', ax=ax)
    upper_band.plot(label='upper band', ax=ax)
    lower_band.plot(label='lower band', ax=ax)    
    rm.plot(label='m20', ax=ax)  

    
    