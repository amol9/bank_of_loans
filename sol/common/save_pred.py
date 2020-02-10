import pandas as pd
from .misc import *

def save(pred, actl, filepath):
    brk()
    result = pd.DataFrame({
        "Predicted": pred,
        "Actual": actl
    })

    result.to_csv(filepath, index=False)
    print("result saved to " + filepath)