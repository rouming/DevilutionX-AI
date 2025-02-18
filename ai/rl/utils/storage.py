import csv
import os
import torch
import logging
import sys

import rl.utils as utils
from .other import device


def get_models_dir():
    return "models"


def get_run_dir(run_name):
    return os.path.join(get_models_dir(), "active", run_name)


def get_status_path(model_dir, best=False):
    if best:
        return os.path.join(model_dir, "best-status.pt")
    return os.path.join(model_dir, "status.pt")


def get_status(model_dir, best=False):
    path = get_status_path(model_dir, best=best)
    return torch.load(path, map_location=device, weights_only=False)


def save_status(status, model_dir):
    path = get_status_path(model_dir)
    torch.save(status, path)


def get_vocab(model_dir):
    return get_status(model_dir)["vocab"]


def get_model_state(model_dir, best=False):
    return get_status(model_dir, best=best)["model_state"]


def get_txt_logger(model_dir):
    path = os.path.join(model_dir, "log.txt")

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger()


def get_csv_logger(model_dir):
    csv_path = os.path.join(model_dir, "log.csv")
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)
