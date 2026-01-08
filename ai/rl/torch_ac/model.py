from abc import abstractmethod, abstractproperty
import torch.nn as nn
import torch.nn.functional as F

class BaseACModel:
    recurrent = False
    num_levels = 1

    @abstractmethod
    def __init__(self, obs_space, action_space):
        pass

    @abstractmethod
    def forward(self, obs, **kwargs):
        pass

    @abstractmethod
    def load_from_status(self, status, logger=None):
        pass

    @abstractmethod
    def save_to_status(self, status):
        pass

class RecurrentACModel(BaseACModel):
    recurrent = True
    num_levels = 1

    @abstractmethod
    def forward(self, obs, memory, **kwargs):
        pass

    @property
    @abstractmethod
    def memory_size(self):
        pass
