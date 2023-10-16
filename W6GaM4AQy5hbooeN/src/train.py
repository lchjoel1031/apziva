import numpy as np
import pandas as pd
from sklearn import metrics, ensemble
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler


import pickle
import matplotlib.pyplot as plt


import os
import warnings
warnings.filterwarnings('ignore')

#os.system('mkdir plot')
os.system('mkdir model')
os.system('mkdir data')


#data preprocessing
data = '../data/term-deposit-marketing-2020.csv'
model = '../model/'
df = pd.read_csv(data)
df.head()

#ordinal encoding
df[['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'y']] = df[['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'y']].apply(LabelEncoder().fit_transform)
df.head()

#balacing
x=df[['age', 'job', 'marital', 'education', 'default', 'balance',
       'housing', 'loan', 'contact', 'day', 'month', 'duration',
       'campaign']].values
y=df.y.values

rus = RandomUnderSampler(random_state=42)

x_rus, y_rus = rus.fit_resample(x, y)

#preparing training data
random_state=69
x_train, x_test, y_train, y_test = train_test_split(x_rus, y_rus, test_size=0.2, random_state=random_state)

features=['age', 'job', 'marital', 'education', 'default', 'balance',
       'housing', 'loan', 'contact', 'day', 'month', 'duration',
       'campaign', ]

#training with grid search
xgb_clas = XGBClassifier(random_state=random_state)

params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

xgb_cv = GridSearchCV(estimator=xgb_clas, param_grid=params, cv=5)
xgb_cv.fit(x_train, y_train)

#save model
with open(model+"xgb_cv_model.pkl", 'wb') as f:
            pickle.dump(
                xgb_cv.best_estimator_,
                f,
                pickle.HIGHEST_PROTOCOL)
