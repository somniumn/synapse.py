import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

font_cs = {'fontname': 'Consolas'}


def pdf(x_, m, v):  # normal distribution or probability density function(pdf)
    mu = m
    sigma = np.sqrt(v)
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * np.power(x_ - mu, 2) / np.power(sigma, 2))


x = np.arange(-10.0, 10.0, 0.1)
mean = np.average(x)
var = 3  # np.var(x)
y = pdf(x, mean, var)
d = np.c_[x, y]

df = pd.DataFrame(d, columns=['x', 'y'])
print(df)

sns.set_theme(style='whitegrid')

sns.lineplot(data=df, x='x', y='y')

plt.xlabel('x', **font_cs)
plt.ylabel('value', **font_cs)
plt.title('Probability Density Function', **font_cs)
plt.show()

