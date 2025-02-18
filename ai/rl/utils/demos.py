import os
import pickle

from .. import utils

def get_demos_path(model_dir, env=None, valid=False):
    valid_suff = '_valid' if valid else ''
    demos_path = ("demos" + valid_suff) + '.pkl'
    return os.path.join(model_dir, demos_path)


def load_demos(path):
    try:
        return pickle.load(open(path, "rb"))
    except FileNotFoundError:
        return []


def save_demos(demos, path):
    pickle.dump(demos, open(path, "wb"))


def transform_demos(demos):
    # Nothing to transform
    return demos
