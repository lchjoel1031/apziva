import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

with open('../model/xgb_cv_model.pkl', 'rb') as f:
    model = pickle.load(f)


def predict(model, data):
    predictions = model.predict(data)
    return predictions



data = pd.read_csv("../data/term-deposit-marketing-2020.csv")
data[['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'y']] = data[['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'y']].apply(LabelEncoder().fit_transform)
data.drop('y', axis=1, inplace=True)



predictions = predict(model, data)
predictions
