from os import path
from os import mkdir
import sys


def make_pred_dirs(exp):
    base = "pred"

    paths = [
        "",
        "test",
        "train",
        "models",
        "result",
        "other",
        "models\\tf",
        "models\\sk",
        "models\\xg",
        "result\\tf",
        "result\\sk",
        "result\\xg"
    ]

    for p in paths:
        fp = path.join("pred", exp, p)
        if not path.exists(fp):
            mkdir(fp)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exp = sys.argv[1]
        make_pred_dirs(exp)
    else:
        print("please provide experiment name as command line argument")

