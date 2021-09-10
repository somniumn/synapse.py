from sklearn.model_selection import train_test_split


# data split
def divide_dataset(dataset, label_name, train_rate=0.70, random_seed=None):
    _x = dataset.iloc[:, 0:dataset.shape[1]-1]
    _y = dataset[label_name]
    _x_train, _x_test, _y_train, _y_test = train_test_split(_x, _y, test_size=(1-train_rate)
                                                            , random_state=random_seed
                                                            , shuffle=True)

    # Check dimensions of data after splitting
    print(f"X_train dimensions: {_x_train.shape}")
    print(f"y_train dimensions: {_y_train.shape}\n")

    print(f"X_test dimensions: {_x_test.shape}")
    print(f"y_test dimensions: {_y_test.shape}\n")

    return _x_train, _x_test, _y_train, _y_test


def divide_dataset_with_unlabeled(dataset, label_name, train_rate=0.05, unlabeled_rate=0.70, random_seed=None):
    # Shuffle the data
    dataset = dataset.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    # Generate indices for splits
    index_test = round(len(dataset) * (1.0 - train_rate - unlabeled_rate))
    index_train = index_test + round(len(dataset) * train_rate)
    index_unlabeled = index_train + round(len(dataset) * unlabeled_rate)

    # Partition the data
    test_set = dataset.iloc[:index_test]
    train_set = dataset.iloc[index_test:index_train]
    unlabeled_set = dataset.iloc[index_train:index_unlabeled]

    # Assign data to train, test, and unlabeled sets
    _x_train = train_set.drop(label_name, axis=1)
    _y_train = train_set[label_name]

    _x_unlabeled = unlabeled_set.drop(label_name, axis=1)

    _x_test = test_set.drop(label_name, axis=1)
    _y_test = test_set[label_name]

    # Check dimensions of data after splitting
    print(f"\n ========== divided dataset status ========== \n")
    print(f"Total dataset dimensions: {dataset.shape}")
    print(f"X_train dimensions: {_x_train.shape}")
    print(f"y_train dimensions: {_y_train.shape}")

    print(f"X_test dimensions: {_x_test.shape}")
    print(f"y_test dimensions: {_y_test.shape}")

    print(f"X_unlabeled dimensions: {_x_unlabeled.shape}\n")

    return _x_train, _x_test, _x_unlabeled, _y_train, _y_test


def get_feature_names(dataset, label_name):
    feature_names = list(dataset.columns)
    feature_names.remove(label_name)
    return feature_names


