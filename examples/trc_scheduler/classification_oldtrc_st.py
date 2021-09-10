import itertools

from pycomp.viz.insights import plot_corr_matrix, data_overview
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, precision_recall_curve, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, plot_confusion_matrix
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from xgboost import plot_importance

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import *
from keras.layers import *
from sklearn.metrics import accuracy_score
import tensorflow as tf

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
from tensorflow import metrics
import os

# Self Training Model

# functions define


def plot_model_predictions(m):
    xx, yy = np.meshgrid(np.arange(-1.4, 1.4, 0.1),
                         np.arange(-1.8, 1.4, 0.1))

    Z = m.predict(np.c_[xx.ravel(), yy.ravel()]).argmax(-1)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='Greens')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=20, cmap='winter', edgecolor='none', alpha=0.005)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, cmap='winter', edgecolor='k')

    plt.show()


def compute_kld(p_logit, q_logit):
    p = tf.nn.softmax(p_logit)
    q = tf.nn.softmax(q_logit)
    return tf.reduce_sum(p*(tf.math.log(p + 1e-16) - tf.math.log(q + 1e-16)), axis=1)


def make_unit_norm(x):
    return x/(tf.reshape(tf.sqrt(tf.reduce_sum(tf.pow(x, 2.0), axis=1)), [-1, 1]) + 1e-16)


# data load and split training set / test set
DATA_PATH = 'C:/Work/HP/dataset'
TRAIN_FILENAME = 'citrine_nn_200403_trc_dataset.csv'

# Reading training data
data_set = pd.read_csv(os.path.join(DATA_PATH, TRAIN_FILENAME))
data_set.head()

# Shuffle the data
data_set = data_set.sample(frac=1, random_state=42).reset_index(drop=True)

class_names = ['ShouldDo', 'NotNecessary']
data_set = data_set.drop('id', axis=1)
feature_names = list(data_set.columns)
feature_names.remove('stroke')
df = data_set

# Generate indices for splits
test_index = round(len(df)*0.20)
train_index = test_index + round(len(df)*0.05)
unlabeled_index = train_index + round(len(df)*0.75)

# Partition the data
test = df.iloc[:test_index]
train = df.iloc[test_index:train_index]
unlabeled = df.iloc[train_index:unlabeled_index]

# Assign data to train, test, and unlabeled sets
X_train = train.drop('ClassBinary', axis=1)
y_train = train.ClassBinary

X_unlabeled = unlabeled.drop('ClassBinary', axis=1)

X_test = test.drop('ClassBinary', axis=1)
y_test = test.ClassBinary

# Check dimensions of data after splitting
print(f"X_train dimensions: {X_train.shape}")
print(f"y_train dimensions: {y_train.shape}\n")

print(f"X_test dimensions: {X_test.shape}")
print(f"y_test dimensions: {y_test.shape}\n")

print(f"X_unlabeled dimensions: {X_unlabeled.shape}")

# visualize class distribution
y_train.value_counts().plot(kind='bar')
plt.xticks([0, 1], ['ShouldDo', 'NotNecessary'])
plt.ylabel('count')

Y_train_cat = keras.utils.to_categorical(y_train)
Y_test_cat = keras.utils.to_categorical(y_test)
Y_test_arr = y_test.to_numpy()

# Replacing none to NaN
data_set = data_set.replace('none', np.nan)

# Data overview
overview = data_overview(df=data_set)
print(overview)

# DecisionTree Classifier
dtree = DecisionTreeClassifier()
dtree_param_grid = {
    'criterion': ['entropy', 'gini'],
    'max_depth': [3, 5, 10, 20],
    'max_features': np.arange(1, len(feature_names)),
    'class_weight': ['balanced', None],
    'random_state': [42]
}
clf1 = RandomizedSearchCV(dtree, param_distributions=dtree_param_grid, scoring='accuracy', cv=5, verbose=1,
                         n_iter=100, random_state=42, n_jobs=-1)
clf1.fit(X_train, y_train)
dtree_model = clf1.best_estimator_
pred = dtree_model.predict(X_test)
y_hat_test = clf1.predict(X_test)
y_hat_train = clf1.predict(X_train)

train_f1 = f1_score(y_train, y_hat_train)
test_f1 = f1_score(y_test, y_hat_test)

print(f"Train f1 Score: {train_f1}")
print(f"Test f1 Score: {test_f1}")
plot_confusion_matrix(clf1, X_test, y_test, display_labels=class_names, cmap=plt.cm.Blues)
# plt.show()

# Initialize
iterations = 0

# Containers to hold f1_scores and # of pseudo-labels
trains_f1s = []
test_f1s = []
pseudo_labels = []

# Assign value to initiate while loop
high_prob = [1]

# Loop
while len(high_prob) > 0:

    # Fit classifier and make train/test predictions
    clf2 = RandomizedSearchCV(dtree, param_distributions=dtree_param_grid, scoring='accuracy', cv=5, verbose=1,
                             n_iter=100, random_state=42, n_jobs=-1)
    clf2.fit(X_train, y_train)
    y_hat_train = clf2.predict(X_train)
    y_hat_test = clf2.predict(X_test)

    # Calculate and print iteration # and f1 scores, and store f1 scores
    train_f1 = f1_score(y_train, y_hat_train)
    test_f1 = f1_score(y_test, y_hat_test)
    print(f"Iteration {iterations}")
    print(f"Train f1: {train_f1}")
    print(f"Test f1: {test_f1}")
    trains_f1s.append(train_f1)
    test_f1s.append(test_f1)

    # Generate predictions and probabilities for unlabeled data
    print(f"Now predicting labels for unlabeled data...")

    pred_probs = clf2.predict_proba(X_unlabeled)
    preds = clf2.predict(X_unlabeled)
    prob_0 = pred_probs[:, 0]
    prob_1 = pred_probs[:, 1]

    # Store predictions and probabilities in dataframe
    df_pred_prob = pd.DataFrame([])
    df_pred_prob['preds'] = preds
    df_pred_prob['prob_0'] = prob_0
    df_pred_prob['prob_1'] = prob_1
    df_pred_prob.index = X_unlabeled.index

    # Separate predictions with > 90% probability
    high_prob = pd.concat([df_pred_prob.loc[df_pred_prob['prob_0'] > 0.9],
                           df_pred_prob.loc[df_pred_prob['prob_1'] > 0.9]],
                          axis=0)
    print(f"{len(high_prob)} high-probability predictions added to training data.")

    pseudo_labels.append(len(high_prob))

    # Add pseudo-labeled data to training data
    X_train = pd.concat([X_train, X_unlabeled.loc[high_prob.index]], axis=0)
    y_train = pd.concat([y_train, high_prob.preds])

    # Drop pseudo-labeled instances from unlabeled data
    X_unlabeled = X_unlabeled.drop(index=high_prob.index)
    print(f"{len(X_unlabeled)} unlabeled instances remaining.\n")

    # Update iteration counter
    iterations += 1

    if len(X_unlabeled) == 0:
        break


# Plot f1 scores and number of pseudo-labels added for all iterations
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6,8))
ax1.plot(range(iterations), test_f1s)
ax1.set_ylabel('f1 Score')
ax2.bar(x=range(iterations), height=pseudo_labels)
ax2.set_ylabel('Pseudo-Labels Created')
ax2.set_xlabel('# Iterations')

# View confusion matrix after self-training
plot_confusion_matrix(clf2, X_test, y_test, cmap='Blues', display_labels=class_names)
plt.show()

