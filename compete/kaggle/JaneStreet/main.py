# numpy
import numpy as np

# pandas stuff
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# plotting stuff
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
colorMap = sns.light_palette("blue", as_cmap=True)

# install stuff
# !pip install dabl > /dev/null
import dabl
# install datatable
# !pip install datatable > /dev/null
import datatable as dt

# misc
import missingno as msno

# system
import warnings
warnings.filterwarnings('ignore')
# for the image import
import os
from IPython.display import Image
# garbage collector to keep RAM in check
import gc
import time


pd.set_option("display.max_rows", None, "display.max_columns", None)
# Read data
start_time = time.time()
# train_data_datatable = dt.fread('C:/Work/Kaggle/dataset/jane-street-market-prediction/train.csv')
train_data = pd.read_csv('C:/Work/Kaggle/dataset/jane-street-market-prediction/train.csv')
print(time.time() - start_time)

fig, ax = plt.subplots(figsize=(15, 5))
balance = pd.Series(train_data['resp']).cumsum()
ax.set_xlabel("Trade", fontsize=18)
ax.set_ylabel("Cumulative return", fontsize=18)
balance.plot(lw=3)

fig, ax = plt.subplots(figsize=(15, 5))
balance = pd.Series(train_data['resp']).cumsum()
resp_1 = pd.Series(train_data['resp_1']).cumsum()
resp_2 = pd.Series(train_data['resp_2']).cumsum()
resp_3 = pd.Series(train_data['resp_3']).cumsum()
resp_4 = pd.Series(train_data['resp_4']).cumsum()
ax.set_xlabel("Trade", fontsize=18)
ax.set_title("Cumulative return of resp and time horizons 1, 2, 3, and 4 (500 days)", fontsize=18)
balance.plot(lw=3)
resp_1.plot(lw=3)
resp_2.plot(lw=3)
resp_3.plot(lw=3)
resp_4.plot(lw=3)
plt.legend(loc="upper left")

plt.figure(figsize=(12, 5))
ax = sns.distplot(train_data['resp'],
                  bins=3000,
                  kde_kws={"clip":(-0.05,0.05)},
                  hist_kws={"range":(-0.05,0.05)},
                  color='darkcyan',
                  axlabel="Histogram of resp values",
                  kde=False)
values = np.array([rec.get_height() for rec in ax.patches])
norm = plt.Normalize(values.min(), values.max())
colors = plt.cm.jet(norm(values))
for rec, col in zip(ax.patches, colors):
    rec.set_color(col)

plt.show()

