import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier

from ..common.common_cols import *
from ..common.transform import *
from .test import *
from ..common import config
from ..common.misc import *


def logistic_rg():
    dataset = pd.read_csv(config.train_fn)
    dataset = transform_dataset(dataset)

    com_cols = load_common_cols()

    x_train = dataset[com_cols]
    y_train = dataset[['loan_status']]

    #dataset = np.matrix(pd.read_csv(".\\data\\10k_train.csv", header=1).values)

    #x_train = np.matrix(x_train.values).transpose()
    #y_train = np.matrix(y_train.values).transpose()

    model = XGBClassifier()
    model.fit(x_train, y_train.values.ravel())

    pred(model, lg=True)
