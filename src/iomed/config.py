# coding=utf-8
""" module defining config file i/o """
import json
import yaml
from collections import abc
from pathlib import Path
import numpy as np


def read(path):
    """read config file"""
    path = Path(path)
    if not path.suffix or path.suffix == ".json":
        # json
        path = path.with_suffix(".json")
        with open(path, "r") as fp:
            return json.load(fp, object_hook=json_deserializer)

    elif path.suffix == ".yml":
        # ymal
        with open(path, "r") as fp:
            return yaml.load(fp, Loader=yaml.Loader)
    else:
        raise ValueError(f"Unknown config file type: {path}")


def write(path, config, kind="json"):
    """write config file"""
    path = Path(path)
    config = tame(config)
    if path.suffix:
        kind = path.suffix[1:]

    if kind == "json":
        path = path.with_suffix(".json")
        with open(path, "w") as fp:
            json.dump(config, fp, default=json_serializer)

    elif kind in ["yaml", "yml"]:
        path = path.with_suffix(".yml")
        with open(path, "w") as fp:
            yaml.dump(config, fp, default_flow_style=False)

    else:
        raise ValueError(f"Unknown config file type: {kind}")


def tame(element):
    """convert dct elements to standard list/str/numbers"""
    if isinstance(element, (str, type(None))):
        return element
    elif isinstance(element, abc.Mapping):
        return {tame(key): tame(value) for key, value in element.items()}
    elif isinstance(element, (np.ndarray, abc.Sequence)):
        return [tame(item) for item in element]
    elif isinstance(element, (int, np.integer)):
        return int(element)
    elif isinstance(element, (np.floating, float)):
        return float(element)
    elif np.iscomplex(element):
        return complex(element)
    else:
        return str(element)


# JSON I/O
def json_serializer(obj):
    if isinstance(obj, np.ndarray):
        return {"ndarray": obj.tolist()}
    return obj


def json_deserializer(dct):
    if "ndarray" in dct:
        return np.asarray(dct["ndarray"])
    return dct


# YAML I/O
def numpy_representer(dumper, data):
    return dumper.represent_sequence("!ndarray", data.tolist())


def numpy_constructor(loader, node):
    seq = loader.construct_sequence(node)
    return np.asarray(seq)


yaml.add_representer(np.ndarray, numpy_representer)
yaml.add_constructor("!ndarray", numpy_constructor)
