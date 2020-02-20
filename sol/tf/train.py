import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

from ..common.common_cols import *
from ..common.transform import *
from .test import *
from ..common import config
from ..common.misc import *


def linear_rg():
    dataset = pd.read_csv(config.train_fn)
    dataset = transform_dataset(dataset)

    com_cols = load_common_cols()

    x_train = dataset[com_cols]
    y_train = dataset[['loan_status']]

    #dataset = np.matrix(pd.read_csv(".\\data\\10k_train.csv", header=1).values)

    x_train = np.matrix(x_train.values).transpose()
    y_train = np.matrix(y_train.values).transpose()

    n = x_train.shape[0]

    x = tf.placeholder(tf.float32, shape=(n, None))
    y = tf.placeholder(tf.float32, shape=(1, None))

    A = tf.get_variable("A", shape=(1, n))
    b = tf.get_variable("b", shape=())

    f = tf.matmul(A, x) + b

    L = tf.reduce_sum((f - y) ** 2)

    optimizer = None
    if linear_rg:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(L)
    else:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(L)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    runs = config.tf.runs
    for t in range(runs):
        _, current_loss, current_A, current_b = session.run([optimizer, L, A, b], feed_dict={ x: x_train, y: y_train})
        print("t = %g, loss = %g, A = %s, b = %g"%(t, current_loss, str(current_A)[0:30], current_b))


    #with open("model_tf", "wb") as fl:
    #    pickle.dump(f, fl)

    pred(f, session, x)


def logistic_rg():
    dataset = pd.read_csv(config.train_fn)
    dataset = transform_dataset(dataset)

    com_cols = load_common_cols()

    x_train = dataset[com_cols]
    y_train = dataset[['loan_status']]

    #dataset = np.matrix(pd.read_csv(".\\data\\10k_train.csv", header=1).values)

    x_train = np.matrix(x_train.values).transpose()
    y_train = np.matrix(y_train.values).transpose()

    n = x_train.shape[0]

    x = tf.placeholder(tf.float32, shape=(n, None))
    y = tf.placeholder(tf.float32, shape=(1, None))

    A = tf.get_variable("A", shape=(1, n))
    b = tf.get_variable("b", shape=())

    f = tf.matmul(A, x) + b

    L = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f, labels=y))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(L)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    runs = config.tf.runs
    for t in range(runs):
        _, current_loss, current_A, current_b = session.run([optimizer, L, A, b], feed_dict={ x: x_train, y: y_train})
        print("t = %g, loss = %g, A = %s, b = %g"%(t, current_loss, str(current_A)[0:30], current_b))


    #with open("model_tf", "wb") as fl:
    #    pickle.dump(f, fl)

    pred(f, session, x, lg=True)  



