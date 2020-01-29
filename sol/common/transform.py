import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import math


def transform_dataset(dataset):
    dataset = dataset.fillna(0)


    #scrub
    def transform_term():
        t = dataset['term']
        t.transform(lambda x: int(x.split()[0]))
        dataset['term'] = t

    def transform(col, f):
        l = len(dataset[col])
        for i in range(0, l):
            dataset[col][i] = f(dataset[col][i])

    def transform_lwp(v):
        if type(v) != str and math.isnan(v):   #??
            return 0

        if v == 0:
            return 0

        m = re.search("\\d+", v)
        if m is not None:
            return m[0]
        else:
            return 0

    def transform_vsj(v):
        if type(v) != str and math.isnan(v):
            return 0
        elif v.lower() == 'verified':
            return 1
        else:
            return 2

    def transform_at(v):
        if v == "INDIVIDUAL":
            return 0
        else:
            return 1


    #transform_term()
    dataset['term'] = dataset['term'].transform(lambda x: int(x.split()[0]))
    dataset['last_week_pay'] = dataset['last_week_pay'].transform(transform_lwp).astype('int32')
    #dataset['verification_status_joint'] = dataset['verification_status_joint'].transform(transform_vsj).astype('int8')
    #dataset['application_type'] = dataset['application_type'].transform(transform_at).astype('int8')
    #dataset['initial_list_status'] = dataset['initial_list_status'].transform(lambda x: 0 if x == 'w' else 1).astype('int8')

    #states = {}
    #for i, s in enumerate(dataset['addr_state'].unique().tolist()): states[s] = i
    #dataset['addr_state'] = dataset['addr_state'].transform(lambda x: states[x]).astype('int8')

    #dataset['zip_code'] = dataset['zip_code'].transform(lambda x: x[0:3]).astype('int32')

    #d = {}
    #col = 'somethi'
    #for i, s in enumerate(dataset['col'].unique().tolist()): states[s] = i

    # emp_length
    dataset['emp_length'] = dataset['emp_length'].transform(transform_lwp).astype('int32')


    dummy_list = [
        'verification_status_joint',
        'application_type',
        'initial_list_status',
        'zip_code',
        'grade',
        'sub_grade',
        'home_ownership',
        'verification_status',
        'pymnt_plan',
        'purpose'
    ]

    ignore_list = [
        'batch_enrolled',
        'emp_title',
        'addr_state',
        'desc',
        'title'
    ]

    dataset = pd.get_dummies(dataset, "d", "_", columns=dummy_list)
    dataset = dataset.drop(columns=ignore_list)

    return dataset

