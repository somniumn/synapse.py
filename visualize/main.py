import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

font_cs = {'fontname': 'Consolas'}


def sigmoid(x):
    return 1 / (1+np.exp(-x))


def pdf(x_, m, v):  # normal distribution or probability density function(pdf)
    mu = m
    sigma = np.sqrt(v)
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * np.power(x_ - mu, 2) / np.power(sigma, 2))


def gaussian(x_, m, v):
    mu = m
    sigma = np.sqrt(v)
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(0.5 * np.power(x_ - mu, 2) / np.power(sigma, 2))


def rbf(x_, m, v):
    mu = m
    sigma = np.sqrt(v)
    return np.exp(-0.5 * np.power(x_-mu, 2) / np.power(sigma, 2))


x = np.arange(-10.0, 10.0, 0.1)

# switch functions
# y = sigmoid(x)
mean = np.average(x)
var = 3  # np.var(x)
y1 = pdf(x, mean, var)
y2 = gaussian(x, mean, var)
y3 = rbf(x, mean, var)

# d = np.c_[x, y1, y2, y3]
# df = pd.DataFrame(d, columns=['X', 'pdf', 'gaussian', 'rbf'])
d = np.c_[y1, y3]
df = pd.DataFrame(d, x, columns=['pdf', 'rbf'])
df.head()
print(df)

sns.set_theme(style='whitegrid')

sns.lineplot(data=df)
# plt.title('Sigmoid(x)')
# sns.displot(df, x='x', y='pdf')

plt.xlabel('input: x', **font_cs)
plt.ylabel('output: value', **font_cs)
plt.title('PDF(x) vs. RBF(x)', **font_cs)
plt.show()

