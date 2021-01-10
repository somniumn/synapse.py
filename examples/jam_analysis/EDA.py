import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
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
# df['Pass / Fail'] = df['Pass / Fail'].replace('none', '0')
df['Pass / Fail'] = df['Pass / Fail'].replace('none', '1')
pass_map = {'1': 'Jam', '0': 'NoJam'}
pass_colors = ['crimson', 'darkslateblue']
plot_donut_chart(df=df, col='Pass / Fail', label_names=pass_map, colors=pass_colors,
                 title='Absolute Total and Percentual of Pass/Fail Results')

print(df)

plt.show()


