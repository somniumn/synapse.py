import itertools
from pycomp.viz.insights import plot_corr_matrix, data_overview
from sklearn.metrics import accuracy_score, precision_score, precision_recall_curve, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

import keras
from keras.models import *
from keras.layers import *
from sklearn.metrics import accuracy_score
import tensorflow as tf

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow import metrics
import os

from dataset.load import load_dataset
from dataset.divide import divide_dataset_with_unlabeled, get_feature_names
# from visualize.plot_features import plot_features

# Self Training Model
# data load and split training set / test set
DATA_PATH = 'C:/Work/HP/dataset'
# FILE_NAME = 'healthcare-dataset-stroke-data-revision.csv'
FILE_NAME = 'surgical-deepnet.csv'
label_name = 'complication'

# Reading training data
data_set, feature_names = load_dataset(DATA_PATH, FILE_NAME, label_name)

# dataset split
# data_set = data_set.drop('id', axis=1)
X_train, X_test, X_unlabeled, y_train, y_test = divide_dataset_with_unlabeled(
    data_set, label_name, train_rate=0.05, unlabeled_rate=0.65)
class_names = ['No Stroke', 'Stroke']
input_num = len(feature_names)
print(feature_names)

# visualize class distribution
y_test.value_counts().plot(kind='bar')
plt.xticks([0, 1], class_names)
plt.ylabel('count')

plt_i = 1
plt_j = 1
plt_n = 1
while input_num > (plt_i * plt_j):
    if plt_i == plt_j:
        plt_i = plt_i + 1
    else:
        plt_j = plt_j + 1


fig, axes = plt.subplots(ncols=plt_i, nrows=plt_j)
fig.tight_layout()
for i in feature_names:
    plt.rc('font', size=8)
    plt.rc('axes', titlesize=8)
    plt.subplot(plt_i, plt_j, plt_n)
    plt.title(f"feature : {plt_n}")
    plt.xlabel(i)
    plt.ylabel('count')

    if data_set[i].dtype == np.int64:
        sns.countplot(data_set[i])
    elif data_set[i].dtype == np.float64:
        sns.boxplot(data_set[i])

    plt_n = plt_n + 1

Y_train_cat = keras.utils.to_categorical(y_train)
Y_test_cat = keras.utils.to_categorical(y_test)
y_test_arr = y_test.to_numpy()

# Replacing none to NaN
data_set = data_set.replace('none', np.nan)

# Data overview
print(data_overview(df=data_set))

# build model as neural-net
net = Sequential()
net.add(Dense(24, activation='relu', input_shape=(input_num,)))
net.add(Dense(48, activation='relu'))
net.add(Dense(2, activation='sigmoid'))

net_input = Input((input_num,))
p_logit = net(net_input)
p = Activation('softmax')(p_logit)
net.summary()
net.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

net.fit(X_train, Y_train_cat, epochs=20)
pred = net.predict(X_test)
y_hat_test = net.predict(X_test).argmax(-1)
y_hat_train = net.predict(X_train).argmax(-1)
y_train_arr = y_train.to_numpy()
y_test_arr = y_test.to_numpy()

train_f1 = f1_score(y_train, y_hat_train)
test_f1 = f1_score(y_test, y_hat_test)

print(f"Train f1 Score: {train_f1}")
print(f"Test f1 Score: {test_f1}")
cfm = confusion_matrix(y_true=y_test_arr, y_pred=y_hat_test, labels=[0, 1])
print(cfm)

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
    net.fit(X_train, Y_train_cat, epochs=20, verbose=2)
    y_hat_train = np.argmax(net.predict(X_train), axis=-1)
    y_hat_test = np.argmax(net.predict(X_test), axis=-1)
    y_train_arr = y_train.to_numpy()
    y_test_arr = y_test.to_numpy()

    # Calculate and print iteration # and f1 scores, and store f1 scores
    train_f1 = f1_score(y_train_arr, y_hat_train)
    test_f1 = f1_score(y_test_arr, y_hat_test)
    test_accuracy = accuracy_score(y_test_arr, y_hat_test)
    test_precision = precision_score(y_test_arr, y_hat_test)
    test_recall = recall_score(y_test_arr, y_hat_test)
    print(f"Iteration {iterations}")
    print(f"Train f1: {train_f1}")
    print(f"Test f1: {test_f1}")
    print(f"Test accuracy: {test_accuracy}")
    print(f"Test precision: {test_precision}")
    print(f"Test recall: {test_recall}")
    trains_f1s.append(train_f1)
    test_f1s.append(test_f1)

    print(f"===============")
    print(y_test_arr)
    print(y_hat_test)
    print(f"===============")

    # Generate predictions and probabilities for unlabeled data
    print(f"Now predicting labels for unlabeled data...")

    # preds = net.predict_classes(X_unlabeled)
    preds = np.argmax(net.predict(X_unlabeled), axis=-1)
    # print(preds)
    pred_probs = net.predict(X_unlabeled)
    # print(pred_probs)
    prob_0 = pred_probs[:, 0]
    prob_1 = pred_probs[:, 1]
    # prob_0 = preds[:, 0]
    # prob_1 = preds[:, 1]

    # Store predictions and probabilities in dataframe
    df_pred_prob = pd.DataFrame([])
    df_pred_prob['preds'] = preds
    df_pred_prob['prob_0'] = prob_0
    df_pred_prob['prob_1'] = prob_1
    df_pred_prob.index = X_unlabeled.index

    # Separate predictions with > 99% probability
    high_prob = pd.concat([df_pred_prob.loc[df_pred_prob['prob_0'] > 0.99],
                           df_pred_prob.loc[df_pred_prob['prob_1'] > 0.99]],
                          axis=0)
    print(f"{len(high_prob)} high-probability predictions added to training data.")

    pseudo_labels.append(len(high_prob))

    # Add pseudo-labeled data to training data
    X_train = pd.concat([X_train, X_unlabeled.loc[high_prob.index]], axis=0)
    y_train = pd.concat([y_train, high_prob.preds])
    Y_train_cat = keras.utils.to_categorical(y_train)

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
# plot_confusion_matrix(clf, X_test, y_test, cmap='Blues', display_labels=class_names)

y_hat_test = np.argmax(net.predict(X_test), axis=-1)
y_test_arr = y_test.to_numpy()
cfm = confusion_matrix(y_true=y_test_arr, y_pred=y_hat_test, labels=[0, 1])
print(cfm)
plt.show()
