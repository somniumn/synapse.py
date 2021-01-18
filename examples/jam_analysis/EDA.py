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

figsize = (10, 7)
fig, ax = plt.subplots(figsize=figsize)
format_spines(ax=ax, right_border=False)

# Jam rate
# df['Pass / Fail'] = df['Pass / Fail'].replace('none', '1')
df['Pass / Fail'] = df['Pass / Fail'].replace('none', '0')
jam_map = {'1': 'Jam', '0': 'NoJam'}
jam_colors = ['crimson', 'darkslateblue']
plot_donut_chart(df=df, col='Pass / Fail', label_names=jam_map, colors=jam_colors,
                 title='Absolute Total and Percentual of Pass/Fail Results')

"""
# Countplot for duplex
duplex_colors = ['lightskyblue', 'lightcoral']
duplex_map = {'0': 'Simplex', '1': 'Duplex'}
plot_countplot(df=df, col='Duplex', palette=duplex_colors, label_names=duplex_map,
               title='Total Jam Case by Duplex')

# Jam rate by duplex
plot_countplot(df=df, col='Pass / Fail', hue='Duplex', label_names=jam_map, palette=duplex_colors,
               title="Could duplex had some influence on Jam Error?")

# Plotting a double donut chart
plot_double_donut_chart(df=df, col1='Pass / Fail', col2='Duplex', label_names_col1=jam_map,
                        colors1=['crimson', 'navy'], colors2=['lightcoral', 'lightskyblue'],
                        title="Did the duplex influence on jam rate?")
"""
"""
# Countplot for Target Bin
# df['Target Bin'] = df['Target Bin'].replace('none', '2')
df['Target Bin'] = df['Target Bin'].replace('none', '1')
target_bin_colors = ['lightskyblue', 'lightcoral']
target_bin_map = {'2': 'Target Bin:2', '1': 'Target Bin:else'}
plot_countplot(df=df, col='Target Bin', palette=target_bin_colors, label_names=target_bin_map,
               title='Total Jam Case by Target Bin')

# Jam rate by Target Bin
plot_countplot(df=df, col='Pass / Fail', hue='Target Bin', label_names=jam_map, palette=target_bin_colors,
               title="Could Target Bin had some influence on Jam Error?")
"""
"""
# Countplot for Error Code
# df['Error Code'] = df['Error Code'].replace('none', '703')
df['Error Code'] = df['Error Code'].replace('none', '0')
error_code_colors = ['lightskyblue', 'lightcoral']
error_code_map = {'703': 'Error:703', '0': 'Else'}
plot_countplot(df=df, col='Error Code', palette=error_code_colors, label_names=error_code_map,
               title='Total Jam Case by Error Code')

# Jam rate by Error Code
plot_countplot(df=df, col='Pass / Fail', hue='Error Code', label_names=jam_map, palette=error_code_colors,
               title="Could Error Code had some influence on Jam Error?")
"""

"""
# Countplot for Eos >> ID
# df['Eos >> ID'] = df['Eos >> ID'].replace('none', '0')
# df['Eos >> ID'] = df['Eos >> ID'].replace('none', '1')
df['Eos >> ID'] = df['Eos >> ID'].replace('none', '2')
eos_id_colors = ['lightskyblue', 'lightcoral', 'navy']
eos_id_map = {'0': '0', '1': '1', '2': '2'}
plot_countplot(df=df, col='Eos >> ID', palette=eos_id_colors, label_names=eos_id_map,
               title='Total Jam Case by Eos >> ID')

# Jam rate by Error Code
plot_countplot(df=df, col='Pass / Fail', hue='Eos >> ID', label_names=jam_map, palette=eos_id_colors,
               title="Could Eos >> ID had some influence on Jam Error?")
"""

"""
# Countplot for EosFull
eosfull_colors = ['lightskyblue', 'lightcoral']
eosfull_map = {'10': '10', '20': '20'}
plot_countplot(df=df, col='Eos - Full', palette=eosfull_colors, label_names=eosfull_map,
               title='Total Jam Case by Eos - Full')

# Jam rate by Eos - Full
plot_countplot(df=df, col='Pass / Fail', hue='Eos - Full', label_names=jam_map, palette=eosfull_colors,
               title="Could duplex had some influence on Jam Error?")

# Plotting a double donut chart
plot_double_donut_chart(df=df, col1='Pass / Fail', col2='Eos - Full', label_names_col1=jam_map,
                        colors1=['crimson', 'navy'], colors2=['lightcoral', 'lightskyblue'],
                        title="Did the Eos - Full influence on jam rate?")
"""

"""
# Distribution of EosVal variable
plot_distplot(df=df, col='EosVal', title="EosVal Distribution")

# plot_distplot(df=df, col='EosVal', hue='Pass / Fail', kind='kde',
#              title="Is there any relationship between EosVal distribution\n from jam and no jam case?")
ax.set_title("Is there any relationship between EosVal distribution\n from jam and no jam case?", size=16)
sns.kdeplot(df[df['Pass / Fail']=='0']['EosVal'], ax=ax, color='b', shade=True, Label='0')
sns.kdeplot(df[df['Pass / Fail']=='1']['EosVal'], ax=ax, color='r', shade=True, Label='1')
"""

"""
# Distribution of EXIT ID variable
feature_name = 'EXIT ID'
df[feature_name] = df[feature_name].replace('none', np.nan)
df[feature_name] = df[feature_name].fillna(df[feature_name].median())
print(f'Median value of {feature_name} : {df[feature_name].median()}')

plot_distplot(df=df, col=feature_name, title="EXIT ID Distribution")

ax.set_title("Is there any relationship between EXIT ID distribution\n from jam and no jam case?", size=16)
sns.kdeplot(df[df['Pass / Fail']=='0'][feature_name], ax=ax, color='b', shade=True, Label='0')
sns.kdeplot(df[df['Pass / Fail']=='1'][feature_name], ax=ax, color='r', shade=True, Label='1')
"""

"""
# Distribution of IDInfo_ID variable
feature_name = 'IDInfo_ID'
df[feature_name] = df[feature_name].replace('none', np.nan)
df[feature_name] = df[feature_name].fillna(df[feature_name].median())
print(f'Median value of {feature_name} : {df[feature_name].median()}')

plot_distplot(df=df, col=feature_name, title="IDInfo_ID Distribution")

ax.set_title("Is there any relationship between IDInfo_ID distribution\n from jam and no jam case?", size=16)
sns.kdeplot(df[df['Pass / Fail']=='0'][feature_name], ax=ax, color='b', shade=True, Label='0')
sns.kdeplot(df[df['Pass / Fail']=='1'][feature_name], ax=ax, color='r', shade=True, Label='1')
"""

"""
# Distribution of ID - Eos variable
feature_name = 'ID - Eos'
df[feature_name] = df[feature_name].replace('none', np.nan)
df[feature_name] = df[feature_name].fillna(df[feature_name].median())
print(f'Median value of {feature_name} : {df[feature_name].median()}')

plot_distplot(df=df, col=feature_name, title="ID - Eos Distribution")

ax.set_title("Is there any relationship between ID - Eos distribution\n from jam and no jam case?", size=16)
sns.kdeplot(df[df['Pass / Fail']=='0'][feature_name], ax=ax, color='b', shade=True, Label='0')
sns.kdeplot(df[df['Pass / Fail']=='1'][feature_name], ax=ax, color='r', shade=True, Label='1')
"""

"""
# Distribution of Exit - ID variable
feature_name = 'Exit - ID'
df[feature_name] = df[feature_name].replace('none', np.nan)
df[feature_name] = df[feature_name].fillna(df[feature_name].median())
print(f'Median value of {feature_name} : {df[feature_name].median()}')

plot_distplot(df=df, col=feature_name, title="Exit - ID Distribution")

ax.set_title("Is there any relationship between Exit - ID distribution\n from jam and no jam case?", size=16)
sns.kdeplot(df[df['Pass / Fail']=='0'][feature_name], ax=ax, color='b', shade=True, Label='0')
sns.kdeplot(df[df['Pass / Fail']=='1'][feature_name], ax=ax, color='r', shade=True, Label='1')
"""


# Distribution of Del - Exit variable
feature_name = 'Del - Exit'
df[feature_name] = df[feature_name].replace('none', np.nan)
df[feature_name] = df[feature_name].fillna(df[feature_name].median())
print(f'Median value of {feature_name} : {df[feature_name].median()}')

plot_distplot(df=df, col=feature_name, title="Del - Exit Distribution")

ax.set_title("Is there any relationship between Del - Exit distribution\n from jam and no jam case?", size=16)
sns.kdeplot(df[df['Pass / Fail']=='0'][feature_name], ax=ax, color='b', shade=True, Label='0')
sns.kdeplot(df[df['Pass / Fail']=='1'][feature_name], ax=ax, color='r', shade=True, Label='1')



# print(df[df['Pass / Fail']=='0'][feature_name].count())
# pd.set_option('display.max_rows', 500)
# print(df[df['Pass / Fail']=='0'][feature_name])

plt.show()


