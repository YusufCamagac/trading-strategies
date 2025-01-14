import talib
import numpy as np
import pandas as pd

# Gerçek veri yükle
data_path = 'trading_bot/data/BTCUSD_1h_Coinbase.csv'  # Veri dosyasının yolunu güncelle
data = pd.read_csv(data_path, parse_dates=['datetime'])
data.columns = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
data.set_index('datetime', inplace=True)

# Kapanış fiyatlarını al
close = data['Close'].values

# SMA hesapla
sma = talib.SMA(close, timeperiod=10)

print(sma)