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

from transform import *
from test import *
from common_cols import *

dataset = pd.read_csv("./data/10k_train.csv")#train_indessa.csv")
#dataset = pd.read_csv("./data/train_indessa.csv")
dataset = transform_dataset(dataset)

#transform('last_week_pay', lambda x: int(re.search("\\d+", str(x))[0]))

common_cols = load_common_cols()

x_train = dataset[common_cols].values
y_train = dataset['loan_status'].values

regressor = LinearRegression()
regressor.fit(x_train, y_train)

with open('model', 'wb') as f:
    pickle.dump(regressor, f)

#pred(regressor, train_cols)