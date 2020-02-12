import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

from ..common.transform import *
from ..common.common_cols import *
from ..common import config
from ..common.save_pred import save



def pred(f, session, x, lg=False):
    dataset = pd.read_csv(config.test_fn)

    dataset = transform_dataset(dataset)

    common_cols = load_common_cols()

    y_actl = dataset['loan_status']

    dataset = dataset[common_cols]

    x_test = np.matrix(dataset.values).transpose()


    if f is None:

        with open('model_tf', 'rb') as fl:
            f = pickle.load(fl)


    #session = tf.Session()
    #session.run(tf.global_variables_initializer())

    y_pred = session.run(tf.round(tf.sigmoid(f)), feed_dict={ x: x_test})


    #print(y_pred)
    #with open(config.tf.result_fn, "w") as fl:
    #    fl.write("\n".join(list(map(lambda x: str(x), y_pred.flatten().tolist()))))

    save(y_pred.flatten().astype(int), y_actl.values, config.tf.result_fn, config.tf.acc_fn, lg=lg)

if __name__ == "__main__":
    pred(None)
