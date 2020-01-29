import pandas as pd
import numpy as np
import math

from transform import *

#dataset = pd.read_csv("./data/train_indessa.csv")
dataset = pd.read_csv("./data/test_indessa.csv")

#dataset = dataset.fillna(method='ffill')


def find():
    for c in dataset.columns:
        print(c)
        values = dataset[c].values.flatten().tolist()
        for v in values:
            s = str(v)
            if s.find('36 months') > -1:
                print(c, v)

def transform_term():
    t = dataset['term']
    t.transform(lambda x: int(x.split()[0]))

def check_float_cols():
    for col, typ in dataset.dtypes.to_dict().items():
        #print(col, typ)
        blank = False
        nan = False
        string = False

        #if typ == np.dtype('float64'):
        #    print(col)

        for i in dataset[col]:
            if type(i) == str:
                string = True
            elif i == '':
                blank =True
            elif math.isnan(i):
                nan = True
            
        if string:
            print(col, "strings")

        if blank:
            print(col, "blnaks")

        if nan:
            print(col, "nans")
                    

def transform(col, f):
    l = len(dataset[col])
    for i in range(0, l):
        dataset[col][i] = f(dataset[col][i])

dataset = transform_dataset(dataset)
check_float_cols()