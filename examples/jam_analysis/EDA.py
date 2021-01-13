import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from matplotlib import pyplot as plt
import seaborn as sns
import os
from pycomp.viz.insights import *

# Project variables
DATA_PATH = 'C:/Work/HP/dataset'
TRAIN_FILENAME = 'Fin_H2-6703.csv'
# TEST_FILENAME = 'test.csv'

# Reading training data
df = pd.read_csv(os.path.join(DATA_PATH, TRAIN_FILENAME))
df.head()

# Pass rate
df['Pass / Fail'] = df['Pass / Fail'].replace('none', '0')
# df['Pass / Fail'] = df['Pass / Fail'].replace('none', '1')
pass_map = {'1': 'Jam', '0': 'NoJam'}
pass_colors = ['crimson', 'darkslateblue']
plot_donut_chart(df=df, col='Pass / Fail', label_names=pass_map, colors=pass_colors,
                 title='Absolute Total and Percentual of Pass/Fail Results')

# Countplot for duplex
duplex_colors = ['lightskyblue', 'lightcoral']
duplex_map = {'0': 'Simplex', '1': 'Duplex'}
plot_countplot(df=df, col='Duplex', palette=duplex_colors, label_names=duplex_map,
               title='Total Jam Case by Duplex')

# Jam rate by duplex
plot_countplot(df=df, col='Pass / Fail', hue='Duplex', label_names=pass_map, palette=duplex_colors,
               title="Could duplex had some influence on Jam Error?")

# Plotting a double donut chart
plot_double_donut_chart(df=df, col1='Pass / Fail', col2='Duplex', label_names_col1=pass_map,
                        colors1=['crimson', 'navy'], colors2=['lightcoral', 'lightskyblue'],
                        title="Did the duplex influence on jam rate?")

# Countplot for EosFull
eosfull_colors = ['lightskyblue', 'lightcoral']
eosfull_map = {'10': '10', '20': '20'}
plot_countplot(df=df, col='Eos - Full', palette=eosfull_colors, label_names=eosfull_map,
               title='Total Jam Case by Eos - Full')

# Jam rate by Eos - Full
plot_countplot(df=df, col='Pass / Fail', hue='Eos - Full', label_names=pass_map, palette=eosfull_colors,
               title="Could duplex had some influence on Jam Error?")

# Plotting a double donut chart
plot_double_donut_chart(df=df, col1='Pass / Fail', col2='Eos - Full', label_names_col1=pass_map,
                        colors1=['crimson', 'navy'], colors2=['lightcoral', 'lightskyblue'],
                        title="Did the Eos - Full influence on jam rate?")


# Distribution of EosVal variable
plot_distplot(df=df, col='EosVal', title="EosVal Distribution")

plot_distplot(df=df, col='EosVal', hue='Pass / Fail', kind='kde',
              title="Is there any relationship between EosVal distribution\n from jam and no jam case?")


print(df)

plt.show()


