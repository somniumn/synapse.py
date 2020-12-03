import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

font_cs = {'fontname': 'Consolas'}


def sigmoid(x):
    return 1 / (1+np.exp(-x))


x = np.arange(-10.0, 10.0, 0.1)
y = sigmoid(x)
d = np.c_[x, y]

df = pd.DataFrame(d, columns=['x', 'y'])
print(df)

sns.set_theme(style='whitegrid')

sns.lineplot(data=df, x='x', y='y')

plt.xlabel('x', **font_cs)
plt.ylabel('value', **font_cs)
plt.title('Sigmoid(x)', **font_cs)
plt.show()

