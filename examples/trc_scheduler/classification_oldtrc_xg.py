from pycomp.viz.insights import plot_corr_matrix
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, precision_recall_curve, recall_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, plot_confusion_matrix
import xgboost as xgb
from xgboost import plot_importance

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
from tensorflow import metrics
import os


# data load and split training set / test set
DATA_PATH = 'C:/Work/HP/dataset'
TRAIN_FILENAME = 'citrine_nn_trc_dataset.csv'

# Reading training data
data_set = pd.read_csv(os.path.join(DATA_PATH, TRAIN_FILENAME))
data_set.head()

# print(data_set)
X = data_set.iloc[:, 1:data_set.shape[1]-1]
y = data_set['ClassBinary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

class_names = ['NotNecessary', 'ShouldDo']


# print(X_train)

# pca analysis
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_train)
# X_train = pd.DataFrame(principalComponents, columns=['feature1', 'feature2'])
X_train_A = X_train.iloc[:, [3]]
X_train_B = X_train.iloc[:, [2]]
X_train_AB = pd.concat([X_train_A, X_train_B], axis=1)
X_train_C = X_train.iloc[:, 1:10]
X_train_C = pd.concat([X_train_C, X_train.iloc[:, [101]]], axis=1)
XY_Train = pd.concat([X_train_C, y_train], axis=1)
print(XY_Train)

plt.figure(1)
# corr = X_train_AB.corr(method='pearson').style.background_gradient(cmap='coolwarm', axis=None).set_precision(2)
corr = XY_Train.corr(method='pearson')
sns.heatmap(corr, annot=True, cmap='coolwarm')

# Plotting a correlation matrix
plot_corr_matrix(df=XY_Train, corr='positive', corr_col='ClassBinary',
                 title='Top Features - Correlation Positive by ClassBinary')

plt.figure(2)
indicesToKeep = y_train
plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c=indicesToKeep, s=50)
plt.grid()
# plt.show()


X_test_C = X_test.iloc[:, 1:10]
X_test_C = pd.concat([X_test_C, X_test.iloc[:, [101]]], axis=1)
# decision tree
classifier = xgb.XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=4)
# classifier.fit(X_train, y_train)
classifier.fit(X_train_C, y_train)
# predict = classifier.predict(X_test)
predict = classifier.predict(X_test_C)
y_test = pd.Series.to_numpy(y_test)
conf_matrix = confusion_matrix(y_test, predict, labels=[0, 1])
print(conf_matrix)

plot_confusion_matrix(classifier, X_test_C, y_test, display_labels=class_names, cmap=plt.cm.Blues)

print("accuracy_score: {}".format(accuracy_score(y_test, predict)))
print("precision_score: {}".format(precision_score(y_test, predict)))
print("recall_score: {}".format(recall_score(y_test, predict)))
# print("AUC: Area Under Curve: {}".format(roc_auc_score(y_test, predict[:, 1])))

print(y_test)
print(predict)

plot_importance(classifier, max_num_features=10)
# xgb.plot_tree(classifier)
plt.show()

