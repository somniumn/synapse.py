import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
import os

# Project variables
# DATA_PATH = '../input/titanic'
DATA_PATH = 'C:\Work\HP\dataset'
TRAIN_FILENAME = 'Fin_H2-6703.csv'
TEST_FILENAME = 'test.csv'

# Reading training data
df = pd.read_csv(os.path.join(DATA_PATH, TRAIN_FILENAME))
df.head()

X = df.iloc[:, 1:df.shape[1]-1]
y = df['Pass / Fail']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


model = xgb.XGBClassifier(booster='bgtree', silent=True, min_child_weight=10, max_depth=8,
                          gamma=0, nthread=4, colsample_bytree=0.8, colsample_bylevel=0.9,
                          n_estimators=32, objective='multi:softmax', random_state=2)

model.fit(X_train, y_train, eval_set=[], early_stopping_rounds=50)

model.predict(X_test)
