import pandas as pd
import os


# load dataset from file
def load_dataset(path, file_name, label_name, file_type='csv'):
    if file_type == 'csv':
        dataset = pd.read_csv(os.path.join(path, file_name))
    else:
        print(f"Only can handle csv-file")
        exit()

    dataset.head()
    dataset.info()
    features = list(dataset.columns)
    features.remove(label_name)
    return dataset, features
