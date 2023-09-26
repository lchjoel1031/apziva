import numpy as np
import pandas as pd
from sklearn import metrics, ensemble
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

import pickle
import matplotlib.pyplot as plt


import os
import warnings
warnings.filterwarnings('ignore')

os.system('mkdir model')
os.system('mkdir data')

#read dataset
data = '../data/ACME-HappinessSurvey2020.csv'
df = pd.read_csv(data)

#prepare training and test data
random_state=69
x_train, x_test, y_train, y_test = train_test_split(df[["X1", "X2", "X3", "X4", "X5", "X6"]].values,
                                              (df.Y).values, test_size=0.2, random_state=random_state)

#train with XGBoost
xgb_clas = XGBClassifier(random_state=random_state)
xgb_clas.fit(x_train, y_train)

#evaluate
y_true, y_pred = y_test, xgb_clas.predict(x_test)
xgacc = metrics.accuracy_score(y_true, y_pred)
print("XGBoost accuracy: ", xgacc)

#save trained model
with open("../model/xgboost_model.pkl", 'wb') as f:
            pickle.dump(
                xgb_clas,
                f,
                pickle.HIGHEST_PROTOCOL)
