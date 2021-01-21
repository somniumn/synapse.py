"""
GME: Gaussian Mixture Estimation
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def init_k_means(df, no_clusters=2):
    k_means = KMeans(n_clusters=no_clusters).fit(df)
    centroids = k_means.cluster_centers_
    return centroids


class GaussianMixtureEstimator:
    def __init__(self, no_clusters=2, use_kmeans_init=True):
        self.no_clusters = 2
        self.data = []
        self.means = []
        self.variance = []
        self.proportions = []
        self.no_clusters = no_clusters
        self.use_kmeans_init = use_kmeans_init

    def fit(self, df):
        if len(df.columns) > 2:
            return print('error: data should have up to 2-dimension')

        self.data = df
        if self.use_kmeans_init:
            gm = GaussianMixture(n_components=self.no_clusters, random_state=0)
            gm.means_ = init_k_means(df, no_clusters=self.no_clusters)
            print(f'k-means clustering initialize: {gm.means_}')
            gm.fit(df)
        else:
            gm = GaussianMixture(n_components=self.no_clusters, random_state=0).fit(df)
            print(f'clustering by no k-means initializing')
        self.means = gm.means_
        self.variance = gm.covariances_
        self.proportions = gm.weights_
        return self.means, self.variance, self.proportions



