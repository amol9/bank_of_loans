from os import path
from figlib.api import v1 as fl
import pandas as pd

config = fl.load_json_config("split.json")
dataset_path = path.join(config.src_data_dir, config.src_dataset)

dataset = pd.read_csv(dataset_path)

train_ds = dataset.iloc[:config.train]
test_ds = dataset.iloc[config.train:(config.test + config.train)]

print("split dataset into:")
print("train: " + str(train_ds.shape))
print("test: " + str(test_ds.shape))

train_fn = path.join("pred", config.experiment, "train", "train.csv")
#with open(train_fn, 'w') as f:
train_ds.to_csv(train_fn, index=False)

test_fn = path.join("pred", config.experiment, "test", "test.csv")
#with open(test_fn, 'w') as f:
test_ds.to_csv(test_fn, index=False)

print("wrote csv's")
print("checking train csv: " + str(pd.read_csv(train_fn).shape))
print("checking test csv: " + str(pd.read_csv(test_fn).shape))