from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns

# data load and split training set / test set
data_set = pd.read_csv('C:/Work/Kaggle/dataset/hp/citrine_nn_trc_dataset.csv')
X = data_set.iloc[:, 1:data_set.shape[1]-1]
y = data_set['ClassBinary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# print(X_train)

# pca analysis
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_train)
# X_train = pd.DataFrame(principalComponents, columns=['feature1', 'feature2'])
X_train_A = X_train.iloc[:, [3]]
X_train_B = X_train.iloc[:, [2]]
X_train_AB = pd.concat([X_train_A, X_train_B], axis=1)
X_train_C = X_train.iloc[:, 1:9]
print(X_train_C)

plt.figure(1)
# corr = X_train_AB.corr(method='pearson').style.background_gradient(cmap='coolwarm', axis=None).set_precision(2)
corr = X_train_C.corr(method='pearson')
sns.heatmap(corr, annot=True, cmap='coolwarm')


plt.figure(2)
indicesToKeep = y_train
plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c=indicesToKeep, s=50)
plt.grid()
plt.show()


# decision tree



