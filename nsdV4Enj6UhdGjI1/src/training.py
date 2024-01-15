import numpy as np
import pandas as pd
import os
import yfinance as yf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

nvda = yf.Ticker("NVDA")
df = nvda.history(start="2020-01-01")
#df.head()

df = df.asfreq('b')
df = df.ffill()
df.index = df.index.tz_convert(None)

series = df.Close

# Use statsmodels to decompose
period = 7

size = int(len(series) * 0.8)
train, test = series[:size], series[size:size+5]

#exponential smoothing
expo = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=period, initialization_method="estimated")
results_expo = expo.fit()
