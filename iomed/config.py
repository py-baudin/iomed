import yaml
from collections import abc
import numpy as np

def load(obj, **kwargs):
    return Config.load(obj, **kwargs)

def save(file, config):
    config = Config(config)
    config.save(file)


class Config(dict):
    """ Dictionary with load/save method"""

    def save(self, file):
        data = tame(self)
        with open(file, 'w') as fp:
            yaml.dump(data, fp, default_flow_style=False)

    @classmethod
    def load(cls, file, **kwargs):
        with open(file, 'r') as fp:
            data = yaml.load(fp, Loader=yaml.Loader)
        data.update(**kwargs)
        return cls(data)


def tame(element):
    """convert dct elements to standard list/str/numbers"""
    if isinstance(element, str):
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



# YAML I/O
def numpy_representer(dumper, data):
    return dumper.represent_sequence("!ndarray", data.tolist())


def numpy_constructor(loader, node):
    seq = loader.construct_sequence(node)
    return np.asarray(seq)


yaml.add_representer(np.ndarray, numpy_representer)
yaml.add_constructor("!ndarray", numpy_constructor)
