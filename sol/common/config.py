from figlib.api import v1 as fl
from os import path
from datetime import datetime as dt

now_str = dt.now().strftime("%d_%b_%Y_%H_%M_%S")

cfg = fl.load_json_config("config.json")

exp = cfg.experiment
method = cfg.method

exp_dir = path.join(".", "pred", exp)

test_dir = path.join(exp_dir, "test")
test_fn = path.join(test_dir, "test.csv")

train_dir = path.join(exp_dir, "train")
train_fn = path.join(train_dir, "train.csv")

class tf:
    exp = cfg.tf_exp_name
    model_dir = path.join(exp_dir, "models", "tf")
    result_dir = path.join(exp_dir, "result", "tf")
    runs = cfg.tf_runs
    result_fn = path.join(result_dir, exp + "_" + str(runs) + "_runs_" + now_str + ".csv")
    acc_fn = result_fn[0 : -4] + "_acc.txt"

class sk:
    exp = cfg.sk_exp_name
    model_dir = path.join(exp_dir, "models", "sk")
    result_dir = path.join(exp_dir, "result", "sk")
    result_fn = path.join(result_dir, exp + "_" + now_str + ".csv")
    acc_fn = result_fn[0 : -4] + "_acc.txt"
    max_iter = cfg.sk_max_iter

class xg:
    exp = cfg.xg_exp_name
    model_dir = path.join(exp_dir, "models", "xg")
    result_dir = path.join(exp_dir, "result", "xg")
    result_fn = path.join(result_dir, exp + "_" + now_str + ".csv")
    acc_fn = result_fn[0 : -4] + "_acc.txt"

other_dir = path.join(exp_dir, "other")
com_cols_fn = path.join(other_dir, "com_cols.txt")
