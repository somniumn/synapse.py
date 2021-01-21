import pandas as pd
import os
from pysynapse.gme import GaussianMixtureEstimator


# Project variables
DATA_PATH = 'C:/Work/HP/dataset'
TRAIN_FILENAME = 'Fin_H2-6703.csv'
# TEST_FILENAME = 'test.csv'

# Reading training data
df = pd.read_csv(os.path.join(DATA_PATH, TRAIN_FILENAME))
df.head()

df_sample = pd.DataFrame(df, columns=['IDInfo Time'])
df_sample.head()
gme = GaussianMixtureEstimator(no_clusters=3, use_kmeans_init=True)
mean, var, prop = gme.fit(df_sample)

print(mean)
print(var)
print(prop)
