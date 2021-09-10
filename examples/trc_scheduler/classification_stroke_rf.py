import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, plot_confusion_matrix, accuracy_score, precision_score, \
    recall_score
from sklearn.tree import export_graphviz
import os
from pycomp.viz.insights import *
from pycomp.ml.transformers import FiltraColunas
from pycomp.ml.transformers import EliminaDuplicatas
from pycomp.ml.transformers import SplitDados
import graphviz


# Project variables
DATA_PATH = 'C:/Work/HP/dataset'
TRAIN_FILENAME = 'healthcare-dataset-stroke-data-revision.csv'
# TEST_FILENAME = 'test.csv'

# Reading training data
df = pd.read_csv(os.path.join(DATA_PATH, TRAIN_FILENAME))
df.head()

# Label data handling
X = df.iloc[:, 1:df.shape[1]-1]
y = df['ClassBinary']
class_names = ['ShouldDo','NotNecessary']

# Replacing none to NaN
df = df.replace('none', np.nan)

# Data overview
overview = data_overview(df=df)
# print(overview)

# MODEL_FEATURES = ['Process Speed Ratio', 'Environment', 'Outer Temp', 'Page Count', 'Inner Temp',
#                   'Humidity', 'LSU Temp', 'TRC Enable', 'TRC Elapsed Time', 'LSU Temp DIF']
MODEL_FEATURES = ['Page Count', 'Humidity', 'LSU Temp', 'LSU Temp DIF']
X = df[MODEL_FEATURES]

print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70, random_state=42)

# Results
print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of X_test: {X_test.shape}')
print(f'Shape of y_train: {y_train.shape}')
print(f'Shape of y_test: {y_test.shape}')

# New overview
overview = data_overview(df=X_train)
print(overview)

df_prep = pd.concat([X, y], axis=1)

overview = data_overview(df=df_prep)
print(overview)

df_prep.to_csv('prep_dataset.csv')

# Plotting a correlation matrix
plot_corr_matrix(df=df_prep, corr='positive', corr_col='ClassBinary',
                 title='Top Features - Correlation Positive by ClassBinary')

# Generating objects
logreg = LogisticRegression()
dtree = DecisionTreeClassifier()
forest = RandomForestClassifier()

# Logistic Regression
logreg_param_grid = {
    'C': np.linspace(0.1, 10, 20),
    'penalty': ['l1', 'l2'],
    'class_weight': ['balanced', None],
    'random_state': [42],
    'solver': ['liblinear']
}

# Decision Trees
dtree_param_grid = {
    'criterion': ['entropy', 'gini'],
    'max_depth': [3, 5, 10, 20],
    'max_features': np.arange(1, len(MODEL_FEATURES)),
    'class_weight': ['balanced', None],
    'random_state': [42]
}

# Random Forest
forest_param_grid = {
    'bootstrap': [True, False],
    'max_depth': [3, 5, 10, 20, 50],
    'n_estimators': [50, 100, 200, 500],
    'random_state': [42],
    'max_features': ['auto', 'sqrt'],
    'class_weight': ['balanced', None]
}

# Preparing a dictionary of classifiers and its hyperparameters
set_classifiers = {
    'LogisticRegression': {
        'model': logreg,
        'params': logreg_param_grid
    },
    'DecisionTrees': {
        'model': dtree,
        'params': dtree_param_grid
    },
    'RandomForest': {
        'model': forest,
        'params': forest_param_grid
    }
}

# DecisionTree
trainer = RandomizedSearchCV(dtree, param_distributions=dtree_param_grid, scoring='accuracy', cv=5, verbose=1,
                             n_iter=100, random_state=42, n_jobs=-1)
trainer.fit(X_train, y_train)
dtree_model = trainer.best_estimator_

pred = dtree_model.predict(X_test)

# Graphical analysis of performance
plot_confusion_matrix(dtree_model, X_test, y_test, display_labels=class_names, cmap=plt.cm.Blues)

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Python38/Graphviz/bin'

export_graphviz(dtree_model, out_file='tree.dot', class_names=class_names, feature_names=MODEL_FEATURES,
                impurity=True, filled=True)

with open('tree.dot') as file_reader:
    dot_graph = file_reader.read()

dot = graphviz.Source(dot_graph)
dot.render(filename='tree.png')

display(dtree_model)
display(trainer.best_params_)

print("Just Decision Tree")
print("accuracy_score: {}".format(accuracy_score(y_test, pred)))
print("precision_score: {}".format(precision_score(y_test, pred)))
print("recall_score: {}".format(recall_score(y_test, pred)))

# RandomForest
trainer = RandomizedSearchCV(forest, param_distributions=forest_param_grid, scoring='accuracy', cv=5, verbose=1,
                             n_iter=100, random_state=42, n_jobs=-1)

trainer.fit(X_train, y_train)
forest_model = trainer.best_estimator_

predict = forest_model.predict(X_test)

# Graphical analysis of performance
plot_confusion_matrix(forest_model, X_test, y_test, display_labels=class_names, cmap=plt.cm.Blues)

display(forest_model)
display(trainer.best_params_)
# visualize_classifier(forest_model, X_train, y_train)

print("Random Forest")
print("accuracy_score: {}".format(accuracy_score(y_test, predict)))
print("precision_score: {}".format(precision_score(y_test, predict)))
print("recall_score: {}".format(recall_score(y_test, predict)))

plt.show()
