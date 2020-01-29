import pandas as pd
import pickle

from transform import *

def get_common_cols():
    #train_dataset = pd.read_csv("./data/train_indessa.csv")
    train_dataset = pd.read_csv("./data/10k_train.csv")#train_indessa.csv")
    train_dataset = transform_dataset(train_dataset)

    #test_dataset = pd.read_csv("./data/test_indessa.csv")
    test_dataset = pd.read_csv("./data/10k_test.csv")
    test_dataset = transform_dataset(test_dataset)

    train_cols = train_dataset.columns.tolist()
    test_cols = test_dataset.columns.tolist()

    common_cols = list(set(train_cols) & set(test_cols))

    return common_cols

def store_common_cols(cols):
    with open("common_cols_10k", "wb") as f:
        pickle.dump(cols, f)

def load_common_cols():
    with open("common_cols_10k", "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    cols = get_common_cols()
    store_common_cols(cols)