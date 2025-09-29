import csv
import os
import torch
import logging
import sys

import rl.utils as utils
from .other import device


def get_models_dir():
    return "models"


def get_model_dir(model_name):
    return os.path.join(get_models_dir(), "active", model_name)


def get_status_path(model_dir):
    return os.path.join(model_dir, "status.pt")


def get_best_status_path(model_dir):
    return os.path.join(model_dir, "best-status.pt")


def get_status(model_dir):
    path = get_status_path(model_dir)
    return torch.load(path, map_location=device, weights_only=False)


def get_best_status(model_dir):
    path = get_best_status_path(model_dir)
    return torch.load(path, map_location=device, weights_only=False)


def save_status(status, model_dir):
    path = get_status_path(model_dir)
    torch.save(status, path)


def get_vocab(model_dir):
    return get_status(model_dir)["vocab"]


def get_model_state(model_dir):
    return get_status(model_dir)["model_state"]


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
