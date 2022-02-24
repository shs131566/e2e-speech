import os
import re
import yaml

import tensorflow as tf

from typing import List, Union

def is_cloud_path(path: str) -> bool:
    return bool(re.match(r"^[a-z]+://", path))

def preprocess_paths(paths: Union[List[str], str], isdir: bool=False) -> Union[List[str], str]:
    if isinstance(paths, list):
        paths = [path if is_cloud_path(path) else os.path.abspath(os.path.expanduser(path)) for path in paths]
        for path in paths:
            dirpath = path if isdir else os.path.dirname(path)
            if not tf.io.gfile.exists(dirpath):
                tf.io.gfile.makedirs(dirpath)
        return paths
    elif isinstance(paths, str):
        paths = (paths if is_cloud_path(paths) else os.path.abspath(os.path.expanduser(paths)))
        dirpath = paths if isdir else os.path.dirname(paths)
        if not tf.io.gfile.exists(dirpath):
            tf.io.gfile.exists(dirpath)
        return paths
    return None

def load_yaml(path: str) -> dict:
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u"tag:yaml.org,2002:float",
        re.compile(
            u"""^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
          list(u"-+0123456789."),
    )
    with open(path, "r", encoding="utf-8") as file:
        return yaml.load(file, Loader=loader)
