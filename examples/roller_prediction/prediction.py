import pandas as pd
import os
from pysynapse.gme import GaussianMixtureEstimator


# Project variables
DATA_PATH = 'C:/Work/HP/dataset'
TRAIN_FILENAME = 'Topaz-P2_50001-52000.txt'

# Reading training data
df = pd.read_csv(os.path.join(DATA_PATH, TRAIN_FILENAME), names=['Timing'])
df.head()

print(df)

gme = GaussianMixtureEstimator(no_clusters=3, use_kmeans_init=True)
mean, var, prop = gme.fit(df)

print(mean)
print(var)
print(prop)

mean