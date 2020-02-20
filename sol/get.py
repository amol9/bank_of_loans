import os
from os import path
import requests
from figlib.api import v1 as fl
import zipfile


def get_data():
    config = fl.load_json_config("get.json")

    dst_dir = config.dir

    if not path.exists(dst_dir):
        os.mkdir(dst_dir)
    else:
        if len(os.listdir(dst_dir)) > 0:
            print("data already present in", dst_dir)
            return

    url = config.url

    fn = url.split("/")[-1]

    if not path.exists(fn):
        print("downloading data file..")
        res = requests.get(url)
        with open(fn, 'wb') as f:
            f.write(res.content)

        print("file downloaded to %s, size: %d"%(fn, len(res.content)))

    with zipfile.ZipFile(fn, 'r') as z:
        z.extractall(dst_dir)

        print("zip extracted to", dst_dir)
