import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import math
import pickle

import seaborn as seabornInstance

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from ..common.transform import *
from .test import *
from ..common.common_cols import *
from ..common import config


def main():
    dataset = pd.read_csv(config.train_fn)
    dataset = transform_dataset(dataset)

    #transform('last_week_pay', lambda x: int(re.search("\\d+", str(x))[0]))

    common_cols = load_common_cols()

    x_train = dataset[common_cols].values
    y_train = dataset['loan_status'].values

    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    with open('model', 'wb') as f:
        pickle.dump(regressor, f)

    pred(regressor)


if __name__ == "__main__":
    main()
