import pickle
import pandas as pd
with open('model/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)


def predict(model, data):
    predictions = model.predict(data)
    return predictions



data = pd.read_csv("data/ACME-HappinessSurvey2020.csv")
data.drop('Y', axis=1, inplace=True)



predictions = predict(model, data)
predictions
