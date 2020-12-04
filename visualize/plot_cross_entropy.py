import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

font_cs = {'fontname': 'Consolas'}


def pdf(x_, m, v):  # normal distribution or probability density function(pdf)
    mu = m
    sigma = np.sqrt(v)
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * np.power(x_ - mu, 2) / np.power(sigma, 2))


x1 = np.arange(0.01, 1.00, 0.01)
x2 = 1 - x1
h_x1 = -np.log(x1)
h_x2 = -np.log(x2)
h = x1 * h_x1 + x2 * h_x2

df_x1 = pd.DataFrame(h_x1, x1, columns=['E[-logP(x)]'])
df_x2 = pd.DataFrame(h_x2, x1, columns=['E[-logP(1-x)]'])
df = pd.DataFrame(h, x1, columns=['cross-entropy(x)'])
# df = pd.DataFrame(h, x1, columns=['x*-logP(x)+(1-x)*(-logP(1-x))'])
print(df)

sns.set_theme(style='whitegrid')


plt.figure()
sns.lineplot(data=df_x1)
plt.xlabel('x', **font_cs)
plt.ylabel('value', **font_cs)
plt.title('E[-logP(x)]', **font_cs)

plt.figure()
sns.lineplot(data=df_x2)
plt.xlabel('x', **font_cs)
plt.ylabel('value', **font_cs)
plt.title('E[-logP(1-x)]', **font_cs)

plt.figure()
sns.lineplot(data=df)
plt.xlabel('x', **font_cs)
plt.ylabel('value', **font_cs)
plt.title('Cross Entropy', **font_cs)

plt.show()

