Term Deposit Marketing


In this project, I compared different models -- LR, KNN, SVM, NB, DT, RF, AdaBoost, XGBoost -- for classification. The random forest model gives an accuracy of 0.86 and f1 score of 0.86, and the most important features are duration and balance. The feature to focus on is to increase the duration of contact, the longer the better. The segments of customers that are likely to subscribe to termed deposit is the customers who have higher balance. 

The XGBoost model gives a f1 score of 0.87. I then proceed to refine the XGBoost model with GridSearch and 5-fold cross validation. The trained model of XGBoost is saved in model/, the feature importance plot is saved in plot/, a notebook walking through all steps is saved in notebook/, and the source code to run the training is saved in src/train.py, as well as the source code to apply the model to new dataset in src/app.py
