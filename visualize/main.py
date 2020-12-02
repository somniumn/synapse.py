import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns


def sigmoid(x):
    return 1 / (1+np.exp(-x))


def gaussian(x):
    mu = np.average(x)
    sigma = np.var(x)
    return (1 / (2 * np.pi * sigma)) * np.exp(np.power(x-mu, 2) / np.power(sigma, 2))


x_ = np.arange(-5.0, 5.0, 0.1)
y_ = sigmoid(x_)

d = np.c_[x_, y_]

df = pd.DataFrame(d, columns=['X', 'Y'])

sns.lineplot(data=df, x='X', y='Y')

plt.plot(x_, y_)
plt.show()
