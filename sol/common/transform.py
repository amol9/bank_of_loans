import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .misc import *


def transform_dataset(dataset):
    dataset = dataset.fillna(0)


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

    
    #transform_term()
    dataset['term'] = dataset['term'].transform(lambda x: int(x.split()[0]))
    dataset['last_week_pay'] = dataset['last_week_pay'].transform(transform_lwp).astype('int32')
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
        'purpose',
        'term'
    ]

    ignore_list = [
        'batch_enrolled',
        'emp_title',
        'addr_state',
        'desc',
        'title',
        'member_id'
    ]

    standardize_list = [
        'loan_amnt',
        'funded_amnt',
        'funded_amnt_inv',
        'int_rate',
        'emp_length',
        'annual_inc',
        'dti',
        'delinq_2yrs',
        'inq_last_6mths',
        'mths_since_last_delinq',
        'mths_since_last_record',
        'open_acc',
        'pub_rec',
        'revol_bal',
        'revol_util',
        'total_acc',
        'total_rec_int',
        'total_rec_late_fee',
        'recoveries',
        'collection_recovery_fee',
        'collections_12_mths_ex_med',
        'mths_since_last_major_derog',
        'last_week_pay',
        'acc_now_delinq',
        'tot_coll_amt',
        'tot_cur_bal',
        'total_rev_hi_lim'
    ]


    scaler = MinMaxScaler()# StandardScaler()
    for s in standardize_list:
        dataset[s] = scaler.fit_transform(dataset[s].values.reshape(-1, 1)).flatten()

    for d in dummy_list:
        dataset = pd.get_dummies(dataset, d, "_", columns=[d])

    dataset = dataset.drop(columns=ignore_list)

    brk()
    return dataset

