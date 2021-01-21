import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import os
from pycomp.viz.insights import *
from pycomp.ml.transformers import FiltraColunas
from pycomp.ml.transformers import EliminaDuplicatas
from pycomp.ml.transformers import SplitDados

# Project variables
DATA_PATH = 'C:/Work/HP/dataset'
TRAIN_FILENAME = 'Fin_H2-6703.csv'
# TEST_FILENAME = 'test.csv'

# Reading training data
df = pd.read_csv(os.path.join(DATA_PATH, TRAIN_FILENAME))
df.head()

# Label data handling
# df['Pass / Fail'] = df['Pass / Fail'].replace('none', '1')
df['Pass / Fail'] = df['Pass / Fail'].replace('none', '0')
jam_map = {'1': 'NoJam', '0': 'Jam'}

# Replacing none to NaN
df = df.replace('none', np.nan)

# Data overview
overview = data_overview(df=df)
print(overview)

# Initial features
TARGET = 'Pass / Fail'
TO_DROP = ['Page Width', 'Page Length', 'Process Speed']
INITIAL_FEATURES = list(df.drop(TO_DROP, axis=1).columns)

# Applying transformer
# Like a categorical
feature = 'Target Bin'
df[feature] = df[feature].fillna('1')

feature = 'Error Code'
df[feature] = df[feature].fillna('0')

feature = 'Eos >> ID'
# df[feature] = df[feature].fillna('1')
df[feature] = df[feature].fillna('2')

# Like a numerical
feature = 'IDInfo Time'
df[feature] = df[feature].fillna(df[feature].median())

feature = 'Exit Time'
df[feature] = df[feature].fillna(df[feature].median())

feature = 'Delivered Time'
df[feature] = df[feature].fillna(df[feature].median())

feature = 'Fault'
df[feature] = df[feature].fillna(df[feature].median())

feature = 'EXIT ID'
df[feature] = df[feature].fillna(df[feature].median())

feature = 'IDInfo_ID'
df[feature] = df[feature].fillna(df[feature].median())

feature = 'ID - Eos'
df[feature] = df[feature].fillna(df[feature].median())

feature = 'Exit - ID'
df[feature] = df[feature].fillna(df[feature].median())

feature = 'Del - Exit'
df[feature] = df[feature].fillna(df[feature].median())

overview = data_overview(df=df)
print(overview)

# Application transformer
selector = FiltraColunas(features=INITIAL_FEATURES)
df_slct = selector.fit_transform(df)

# Result
print(f'Shape of original dataset: {df.shape}')
print(f'Shape of dataset after selection: {df_slct.shape}')
df_slct.head()

# Applying transformer
dup_dropper = EliminaDuplicatas()
df_nodup = dup_dropper.fit_transform(df_slct)

# Results
print(f'Total of duplicates before: {df_slct.duplicated().sum()}')
print(f'Total of duplicates after: {df_nodup.duplicated().sum()}')

# Applying transformer
splitter = SplitDados(target=TARGET)
X_train, X_test, y_train, y_test = splitter.fit_transform(df_nodup)

# Results
print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of X_test: {X_test.shape}')
print(f'Shape of y_train: {y_train.shape}')
print(f'Shape of y_test: {y_test.shape}')

initial_pipeline = Pipeline([
    ('selector', FiltraColunas(features=INITIAL_FEATURES)),
    ('dup_dropper', EliminaDuplicatas())
])

# Applying this pipeline into the original data
df_prep = initial_pipeline.fit_transform(df)
print(f'Shape of original dataset: {df.shape}')
print(f'Shape of prep dataset: {df_prep.shape}')

# New overview
overview = data_overview(df=X_train)
print(overview)

# And if you separate the dice into a primitive kind
num_features = [col for col, dtype in X_train.dtypes.items() if dtype != 'object']
cat_features = [col for col, dtype in X_train.dtypes.items() if dtype == 'object']

"""
# Straining it DataFrames for testing
X_train_num = X_train[num_features]
X_train_cat = X_train[cat_features]

print(f'Attribute numerical:\n{num_features}')
print(f'\nAttribute categorical:\n{cat_features}')

# Building a complete pipeline
prep_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

# Applying the prep pipeline
X_train_prep = prep_pipeline.fit_transform(X_train)
X_test_prep = prep_pipeline.fit_transform(X_test)
print(f'Shape of X_train_prep: {X_train_prep.shape}')

# Building a final set of features
cat_encoded_features = prep_pipeline.named_transformers_['cat'].named_steps['encoder'].features_after_encoding
MODEL_FEATURES = num_features + cat_encoded_features
print(f'Total features (must be equal to {X_train_prep.shape[1]}): {len(MODEL_FEATURES)}')
"""
# Preparing a final DataFrame after transformation
MODEL_FEATURES = num_features + cat_features
print(f'Total features (must be equal to {X_train.shape[1]}): {len(MODEL_FEATURES)}')
df_prep = pd.DataFrame(X_train, columns=MODEL_FEATURES)
df_prep[TARGET] = y_train.astype('int64')
print(df_prep[TARGET])

overview = data_overview(df=df_prep)
print(overview)

df_prep.to_csv('prep_dataset.csv')

# Plotting a correlation matrix
plot_corr_matrix(df=df_prep, corr='positive', corr_col='Pass / Fail',
                 title='Top Features - Correlation Positive by Pass / Fail')


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


def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()

    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap,
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # fit the estimator
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap, clim=(y.min(), y.max()),
                           zorder=1)

    ax.set(xlim=xlim, ylim=ylim)


from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, plot_confusion_matrix
from sklearn.tree import export_graphviz
import graphviz
from IPython.display import display

class_names = ['NoJam', 'Jam']

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

# RandomForest
trainer = RandomizedSearchCV(forest, param_distributions=forest_param_grid, scoring='accuracy', cv=5, verbose=1,
                             n_iter=50, random_state=42, n_jobs=-1)

trainer.fit(X_train, y_train)
forest_model = trainer.best_estimator_

pred = forest_model.predict(X_test)


# Graphical analysis of performance
plot_confusion_matrix(forest_model, X_test, y_test, display_labels=class_names, cmap=plt.cm.Blues)
"""
export_graphviz(forest_model, out_file='forest.dot', class_names=class_names, feature_names=MODEL_FEATURES,
               impurity=True, filled=True)

with open('forest.dot') as file_reader:
    dot_graph = file_reader.read()

dot = graphviz.Source(dot_graph)
dot.render(filename='forest.png')
"""
display(forest_model)
display(trainer.best_params_)
# visualize_classifier(forest_model, X_train, y_train)

plt.show()

