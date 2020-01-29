from os import path
from os import mkdir
import sys

exp = sys.argv[1]
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
    "result\\tf",
    "result\\sk"
]

for p in paths:
    fp = path.join("pred", exp, p)
    if not path.exists(fp):
        mkdir(fp)