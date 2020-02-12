from .common import config

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
else:
    print("invalid method")
