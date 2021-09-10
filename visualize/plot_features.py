import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def plot_features(dataset, feature_size, feature_names):
    plt_i = 1
    plt_j = 1
    plt_n = 1

    while feature_size > (plt_i * plt_j):
        if plt_i == plt_j:
            plt_i = plt_i + 1
        else:
            plt_j = plt_j + 1

    fig, axes = plt.subplots(ncols=plt_i, nrows=plt_j)
    fig.tight_layout()
    for i in feature_names:
        plt.rc('font', size=8)
        plt.rc('axes', titlesize=8)
        plt.subplot(plt_i, plt_j, plt_n)
        plt.title(f"feature : {plt_n}")
        plt.xlabel(i)
        plt.ylabel('count')

        if dataset[i].dtype == np.int64:
            sns.countplot(dataset[i])
        elif dataset[i].dtype == np.float64:
            sns.boxplot(dataset[i])

        plt_n = plt_n + 1
