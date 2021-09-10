import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dataset.load import load_dataset
from dataset.divide import divide_dataset
from visualize.plot_features import plot_features
import warnings

warnings.filterwarnings('ignore')

DATA_PATH = 'C:/Work/HP/dataset'
FILE_NAME = 'us-economic-spx-dataset-derivatives.csv'
label_name = 'SPX_change'

# Reading training data
data_set, feature_names = load_dataset(DATA_PATH, FILE_NAME, label_name)
data_set = data_set.drop('DATE', axis=1)
feature_names.remove('DATE')
X_train, X_test, y_train, y_test = divide_dataset(data_set, label_name, train_rate=0.70)
input_num = len(feature_names)
cols = list(data_set.columns)
print(cols)
# X_train.info()

plot_features(data_set, input_num, feature_names)

corr = data_set[cols].corr(method='pearson')
# heatmap (seaborn)
fig = plt.figure(figsize=(12, 8))
ax = fig.gca()

sns.set(font_scale=1.0)  # heatmap 안의 font-size 설정
heatmap = sns.heatmap(corr.values, annot=True, fmt='.2f', annot_kws={'size': 12},
                      yticklabels=cols, xticklabels=cols, ax=ax, cmap="RdYlBu")
plt.tight_layout()

# scatter plot
# sns.scatterplot(data=data_set, x='M2SL_change', y=label_name, markers='o', color='blue', alpha=0.6)
# plt.title('Scatter Plot')

from sklearn.preprocessing import StandardScaler

# feature standardization  (numerical_columns except dummy var.-"CHAS")
scaler = StandardScaler()  # 평균 0, 표준편차 1
data_set[feature_names] = scaler.fit_transform(data_set[feature_names])

from sklearn import linear_model
from sklearn import neural_network

# fit regression model in training set
# reg = linear_model.LogisticRegression()
reg = neural_network.MLPRegressor(random_state=1, max_iter=500)
model = reg.fit(X_train, y_train)

# predict in test set
pred_test = reg.predict(X_test)

# 예측 결과 시각화 (test set)
df = pd.DataFrame({'actual': y_test, 'prediction': pred_test})
df = df.sort_values(by='actual').reset_index(drop=True)

plt.figure(figsize=(12, 9))
plt.scatter(df.index, df['prediction'], marker='x', color='r')
plt.scatter(df.index, df['actual'], alpha=0.7, marker='o', color='black')
plt.title("Prediction Result in Test Set", fontsize=20)
plt.legend(['prediction', 'actual'], fontsize=12)

# R square
print(f"R^2 of Training Set: {model.score(X_train, y_train)}")  # training set
print(f"R^2 of Test Set: {model.score(X_test, y_test)}\n")  # test set

# RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt
pred_train = reg.predict(X_train)
print(f"RMSE of Training Set: {sqrt(mean_squared_error(y_train, pred_train))}")  # training set
print(f"RMSE of Test Set: {sqrt(mean_squared_error(y_test, pred_test))}")  # test set

plt.show()



