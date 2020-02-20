import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import math
import pickle

import seaborn as seabornInstance

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import metrics

from ..common.transform import *
from .test import *
from ..common.common_cols import *
from ..common import config


def load():
    dataset = pd.read_csv(config.train_fn)
    dataset = transform_dataset(dataset)

    #transform('last_week_pay', lambda x: int(re.search("\\d+", str(x))[0]))

    common_cols = load_common_cols()

    x_train = dataset[common_cols].values
    y_train = dataset['loan_status'].values

    return x_train, y_train


def linear_rg():
    x_train, y_train = load()

    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    #with open('model', 'wb') as f:
    #    pickle.dump(regressor, f)

    pred(regressor)


def logistic_rg():
    x_train, y_train = load()

    regressor = LogisticRegression(max_iter=config.sk.max_iter)
    regressor.fit(x_train, y_train)

    #with open('model', 'wb') as f:
    #    pickle.dump(regressor, f)

    pred(regressor, lg=True)


if __name__ == "__main__":
    main()
