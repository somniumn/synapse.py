# Importing classifiers
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from pycomp.ml.transformers import FiltraColunas
from pycomp.ml.transformers import EliminaDuplicatas
from pycomp.ml.transformers import SplitDados
from pycomp.ml.transformers import DummiesEncoding
from pycomp.ml.trainer import ClassificadorBinario
from pycomp.viz.insights import *
import os
from warnings import filterwarnings


filterwarnings('ignore')

# Project variables
# DATA_PATH = '../input/titanic'
DATA_PATH = ''
TRAIN_FILENAME = 'train.csv'
TEST_FILENAME = 'test.csv'

# Reading training data
df = pd.read_csv(os.path.join(DATA_PATH, TRAIN_FILENAME))
df.head()

# Initial features
TARGET = 'Survived'
TO_DROP = ['PassengerId', 'Name', 'Ticket', 'Cabin']
INITIAL_FEATURES = list(df.drop(TO_DROP, axis=1).columns)

# Application transformer
selector = FiltraColunas(features=INITIAL_FEATURES)
df_slct = selector.fit_transform(df)

# Applying transformer
dup_dropper = EliminaDuplicatas()
df_nodup = dup_dropper.fit_transform(df_slct)

# Applying transformer
splitter = SplitDados(target='Survived')
X_train, X_test, y_train, y_test = splitter.fit_transform(df_nodup)

# Building initial pipeline
initial_pipeline = Pipeline([
    ('selector', FiltraColunas(features=INITIAL_FEATURES)),
    ('dup_dropper', EliminaDuplicatas()),
    ('splitter', SplitDados(target='Survived'))
])

# Applying pipelines
X_train, X_test, y_train, y_test = initial_pipeline.fit_transform(df)
print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of X_test: {X_test.shape}')
print(f'Shape of y_train: {y_train.shape}')
print(f'Shape of y_test: {y_test.shape}')

# Separando dados por tipo primitivo
num_features = [col for col, dtype in X_train.dtypes.items() if dtype != 'object']
cat_features = [col for col, dtype in X_train.dtypes.items() if dtype == 'object']

# Filtrando DataFrames para testes
X_train_num = X_train[num_features]
X_train_cat = X_train[cat_features]

# Building numerical pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

# Building categorical pipeline
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', DummiesEncoding(dummy_na=False, cat_features_ori=cat_features))
])

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

# Preparing a final DataFrame after transformation
df_prep = pd.DataFrame(X_train_prep, columns=MODEL_FEATURES)
df_prep['Survived'] = y_train

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

# Creating an object and starting training
trainer = ClassificadorBinario()
trainer.fit(set_classifiers, X_train_prep, y_train, random_search=True, scoring='accuracy', cv=5,
            verbose=-1, n_jobs=-1)

# Analytical of training results
metrics = trainer.evaluate_performance(X_train_prep, y_train, X_test_prep, y_test)
print(metrics)

# Graphical analysis of performance
trainer.plot_metrics()

plt.show()


