import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

font_cs = {'fontname': 'Consolas'}


def entropy(x_, b=2):
    if b == 2:
        h_x = -np.sum(x_ * np.log2(x_))
    else:
        h_x = -np.sum(x_ * np.log(x_))
    return h_x


x = np.arange(0.01, 1.00, 0.01)
h = entropy(x)

df = pd.DataFrame(h, x, columns=['entropy(x)'])
# df = pd.DataFrame(h, x1, columns=['x*-logP(x)+(1-x)*(-logP(1-x))'])
print(df)

sns.set_theme(style='whitegrid')

sns.lineplot(data=df)
plt.xlabel('x', **font_cs)
plt.ylabel('value', **font_cs)
plt.title('Entropy', **font_cs)

plt.show()

