from .common import config

if config.method == "tf":
    from .tf import train
    train.main()
elif config.method == "sk":
    from .sk import train
    train.main()
else:
    print("invalid method")
