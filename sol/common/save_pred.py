import pandas as pd
from .misc import *
from .acc import *
from . import config


def save(pred, actl, filepath, acc_filepath):
    result = pd.DataFrame({
        "Predicted": pred,
        "Actual": actl
    })

    result.to_csv(filepath, index=False)
    print("result saved to " + filepath)

    acc_result = calc_accuracy(result)

    with open(acc_filepath, 'w') as f:
        acc_result.save_to_file(f)

    print("accuracy measures saved to " + acc_filepath)
