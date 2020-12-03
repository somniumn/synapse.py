import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

font_cs = {'fontname': 'Consolas'}


def relu(x):
    i = 0
    output = np.zeros(len(x))
    for xi in x:
        if xi < 0:
            output[i] = 0
        else:
            output[i] = xi
        i = i + 1
    return output


x = np.arange(-10.0, 10.0, 0.1)
y = relu(x)
d = np.c_[x, y]

df = pd.DataFrame(d, columns=['x', 'y'])
# print(x)

sns.set_theme(style='whitegrid')

sns.lineplot(data=df, x='x', y='y')

plt.xlabel('x', **font_cs)
plt.ylabel('value', **font_cs)
plt.title('Relu(x)', **font_cs)
plt.show()

