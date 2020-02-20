import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import math
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from ..common.transform import *
from ..common.common_cols import *
from ..common import config
from ..common.save_pred import *


def pred(model, lg=False):
    dataset = pd.read_csv(config.test_fn)

    dataset = transform_dataset(dataset)

    common_cols = load_common_cols()

    y_actl = dataset['loan_status']

    dataset = dataset[common_cols]

    x_test = dataset

    if model is None:
        with open('model', 'rb') as f:
            model = pickle.load(f)

    y_pred = model.predict(x_test)

    save(y_pred.flatten(), y_actl.values, config.sk.result_fn, config.sk.acc_fn, lg=lg)
    #print(y_pred)

if __name__ == "__main__":
    pred(None)
