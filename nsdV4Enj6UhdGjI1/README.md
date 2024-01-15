ValueInvestor:

In this project, I used different time-series forecasting models -- ARIMA, SARIMA, Exponential smoothing, Prophet, LSTM -- to predict stock price. 
The models are evaluated based on RMSE (smaller the better), and the exponential smoothing gives the best result predicting NVIDIA stock price. 
A notebook walking through all steps is saved in notebook/ValueInvestor.ipynb . A comparison of Exponential smoothing method with Bollinger band strategy is in notebook/Comparison with Bollinger band.ipynb .
The source code to run the training is saved in src/train.py, as well as the source code to apply the bert model to new dataset in src/app.py
