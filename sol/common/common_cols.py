import pandas as pd
import pickle

from .transform import *
from . import config

def get_common_cols():
    train_dataset = pd.read_csv(config.train_fn)
    train_dataset = transform_dataset(train_dataset)

    test_dataset = pd.read_csv(config.test_fn)
    test_dataset = transform_dataset(test_dataset)

    train_cols = train_dataset.columns.tolist()
    test_cols = test_dataset.columns.tolist()

    common_cols = list(set(train_cols) & set(test_cols))

    return common_cols

def store_common_cols(cols):
    with open(config.com_cols_fn, "w") as f:
        #pickle.dump(cols, f)
        f.write("\n".join(cols))

def load_common_cols():
    with open(config.com_cols_fn, "r") as f:
        #return pickle.load(f)
        return f.read().splitlines()

if __name__ == "__main__":
    cols = get_common_cols()
    store_common_cols(cols)