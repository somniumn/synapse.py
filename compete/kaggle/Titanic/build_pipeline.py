import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from pycomp.viz.insights import *
import os
from warnings import filterwarnings
from pycomp.ml.transformers import FiltraColunas
from pycomp.ml.transformers import EliminaDuplicatas
from pycomp.ml.transformers import SplitDados
from pycomp.ml.transformers import DummiesEncoding


filterwarnings('ignore')

# Project variables
# DATA_PATH = '../input/titanic'
DATA_PATH = ''
TRAIN_FILENAME = 'train.csv'
TEST_FILENAME = 'test.csv'

# Reading training data
df = pd.read_csv(os.path.join(DATA_PATH, TRAIN_FILENAME))
df.head()

# Data overview
overview = data_overview(df=df)
print(overview)

# Initial features
TARGET = 'Survived'
TO_DROP = ['PassengerId', 'Name', 'Ticket', 'Cabin']
INITIAL_FEATURES = list(df.drop(TO_DROP, axis=1).columns)

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
splitter = SplitDados(target='Survived')
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

# Separando dados por tipo primitivo
num_features = [col for col, dtype in X_train.dtypes.items() if dtype != 'object']
cat_features = [col for col, dtype in X_train.dtypes.items() if dtype == 'object']

# Filtrando DataFrames para testes
X_train_num = X_train[num_features]
X_train_cat = X_train[cat_features]

print(f'Atributos numéricos:\n{num_features}')
print(f'\nAtributos categóricos:\n{cat_features}')

# Applying transformer
imputer = SimpleImputer(strategy='median')
X_train_num_imp = imputer.fit_transform(X_train_num)
X_train_num_imp = pd.DataFrame(X_train_num_imp, columns=num_features)

# Results
print(f'Null data before imputer: {X_train_num.isnull().sum().sum()}')
print(f'Null data after imputer: {X_train_num_imp.isnull().sum().sum()}')

cat_imputer = SimpleImputer(strategy='most_frequent')
X_train_cat_imp = cat_imputer.fit_transform(X_train_cat)
X_train_cat_imp = pd.DataFrame(X_train_cat_imp, columns=cat_features)

# Results
print(f'Null data before imputer: {X_train_cat.isnull().sum().sum()}')
print(f'Null data after imputer: {X_train_cat_imp.isnull().sum().sum()}')


# Creating object and applying transformer
encoder = DummiesEncoding(dummy_na=False)
X_train_cat_enc = encoder.fit_transform(X_train_cat_imp)

# Results
print(f'Shape before encoding: {X_train_cat_imp.shape}')
print(f'Shape after encoding: {X_train_cat_enc.shape}')
X_train_cat_enc.head()


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

# Plotting a correlation matrix
plot_corr_matrix(df=df_prep, corr_col='Survived')

plt.show()


