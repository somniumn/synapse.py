import itertools

from pycomp.viz.insights import plot_corr_matrix, data_overview
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, precision_recall_curve, recall_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, plot_confusion_matrix
import xgboost as xgb
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


# data load and split training set / test set
DATA_PATH = 'C:/Work/HP/dataset'
TRAIN_FILENAME = 'citrine_nn_200403_trc_dataset.csv'

# Reading training data
data_set = pd.read_csv(os.path.join(DATA_PATH, TRAIN_FILENAME))
data_set.head()

# print(data_set)
X = data_set.iloc[:, 1:data_set.shape[1]-1]
y = data_set['ClassBinary']

class_names = ['ShouldDo','NotNecessary']
# MODEL_FEATURES = ['Process Speed Ratio', 'Environment', 'Outer Temp', 'Page Count', 'Inner Temp',
#                   'Humidity', 'LSU Temp', 'TRC Enable', 'TRC Elapsed Time', 'LSU Temp DIF']
MODEL_FEATURES = ['Page Count', 'Humidity', 'LSU Temp', 'LSU Temp DIF']
X = data_set[MODEL_FEATURES]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70, random_state=42)
Y_train_cat = keras.utils.to_categorical(y_train)
Y_test_cat = keras.utils.to_categorical(y_test)
Y_test_arr = y_test.to_numpy()

# Replacing none to NaN
data_set = data_set.replace('none', np.nan)

# Data overview
overview = data_overview(df=data_set)
print(overview)


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


tf.compat.v1.disable_eager_execution()


network = Sequential()
network.add(Dense(100, activation='relu', input_shape=(4,)))
network.add(Dense(2))

model_input = Input((4,))
p_logit = network(model_input)
p = Activation('softmax')(p_logit)

r = tf.random.normal(shape=tf.shape(model_input))
r = make_unit_norm(r)
p_logit_r = network(model_input + 10*r)

kl = tf.reduce_mean(compute_kld(p_logit, p_logit_r))
grad_kl = tf.gradients(kl, [r])[0]
r_vadv = tf.stop_gradient(grad_kl)
r_vadv = make_unit_norm(r_vadv)/3.0

p_logit_no_gradient = tf.stop_gradient(p_logit)
p_logit_r_adv = network(model_input + r_vadv)
vat_loss = tf.reduce_mean(compute_kld(p_logit_no_gradient, p_logit_r_adv))

model_vat = Model(model_input, p)
model_vat.add_loss(vat_loss)

model_vat.compile('sgd', 'categorical_crossentropy', metrics=['accuracy'])

model_vat.metrics_names.append('vat_loss')
model_vat.metrics.append(vat_loss)

# model_vat.fit(np.concatenate([X_train]*10000), np.concatenate([Y_train_cat]*10000))
model_vat.fit(X_train, Y_train_cat)

y_pred = model_vat.predict(X_test).argmax(-1)
print("Test accruracy ", accuracy_score(y_test, y_pred))
print("Precision Score ", precision_score(y_test, y_pred))
print("Recall Score ", recall_score(y_test, y_pred))
print(Y_test_arr)
print(y_pred)
cfm = confusion_matrix(y_true=Y_test_arr, y_pred=y_pred, labels=[0, 1])
print(cfm)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


plot_confusion_matrix(cfm, classes=class_names, normalize=False,  title='Confusion matrix')

# plot_model_predictions(model_vat)
# plot_confusion_matrix(model_vat, X_test, y_test, display_labels=class_names, cmap=plt.cm.Blues)

