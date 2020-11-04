import os
import pandas as pd
import numpy as np


def load_data(root):
    data = pd.DataFrame(columns=['file', 'category'])

    for category in os.listdir(root):
        category_dir = os.path.join(root, category)
        if os.path.isdir(category_dir):
            files = [os.path.join(category_dir, name) for name in os.listdir(category_dir)]
            category_data = pd.DataFrame({'file': files, 'category': [category] * len(files)})
            data = data.append(category_data, ignore_index=True)

    return data


def split_data(data, valid_ratio=0.2):
    train_data = pd.DataFrame(columns=['file', 'category'])
    valid_data = pd.DataFrame(columns=['file', 'category'])

    for _, category_data in data.groupby('category'):
        valid_size = int(valid_ratio * category_data.shape[0])
        valid_index = np.random.choice(category_data.index, size=valid_size, replace=False)
        train_index = np.setdiff1d(category_data.index, valid_index)

        train_data = train_data.append(category_data.loc[train_index], ignore_index=True)
        valid_data = valid_data.append(category_data.loc[valid_index], ignore_index=True)

    return train_data, valid_data


def calculate_weights(data, keywords):
    freqs = data.category.value_counts(normalize=True)
    weights = np.array([freqs[keyword] for keyword in keywords])
    weights = np.concatenate([[1.0 - weights.sum()], weights])
    weights = 1 / weights
    return weights

