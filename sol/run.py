from .common import config

if config.method == "tf":
    from .tf import train
    train.main()
