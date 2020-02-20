from os import path
from figlib.api import v1 as fl
import pandas as pd


def split_data():
    config = fl.load_json_config("split.json")


    train_fn = path.join("pred", config.experiment, "train", "train.csv")
    test_fn = path.join("pred", config.experiment, "test", "test.csv")

    if path.exists(train_fn) and path.exists(test_fn):
        print("no need to split, train and test csv\'s already present")
        return

    dataset_path = path.join(config.src_data_dir, config.src_dataset)
    dataset = pd.read_csv(dataset_path)

    if config.total == "all":
        total = len(dataset)
        if str(config.train).find("*") > -1:
            config.train = int(eval(config.train))
        if str(config.test).find("*") > -1:
            config.test = int(eval(config.test))

        print("calculated: total = %d, train = %d, test = %d"%(total, config.train, config.test))

    train_ds = dataset.iloc[:config.train]
    test_ds = dataset.iloc[config.train:(config.test + config.train)]

    print("split dataset into:")
    print("train: " + str(train_ds.shape))
    print("test: " + str(test_ds.shape))

    #with open(train_fn, 'w') as f:
    train_ds.to_csv(train_fn, index=False)

    #with open(test_fn, 'w') as f:
    test_ds.to_csv(test_fn, index=False)

    print("wrote csv's")
    print("checking train csv: " + str(pd.read_csv(train_fn).shape))
    print("checking test csv: " + str(pd.read_csv(test_fn).shape))
