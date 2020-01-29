from .common import config

if config.lib == "tf":
    from .tf import train
    train.main()
