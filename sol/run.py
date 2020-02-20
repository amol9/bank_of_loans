from .common import config

from .get import get_data
from .common.mk_pred_dirs import make_pred_dirs
from .common.split import split_data
from .common.common_cols import make_common_cols_file

def main():
    get_data()
    make_pred_dirs(config.exp)
    split_data()
    make_common_cols_file()
    run_exp()

def run_exp():
    if config.method == "tf":
        from .tf import train
        train.linear_rg()
    if config.method == "tf.lg":
        from .tf import train
        train.logistic_rg()
    elif config.method == "sk":
        from .sk import train
        train.linear_rg()
    elif config.method == "sk.lg":
        from .sk import train
        train.logistic_rg()
    elif config.method == "xg.lg":
        from .xg import train
        train.logistic_rg()
    else:
        print("invalid method")


if __name__ == "__main__":
    main()

