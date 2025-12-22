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
    def forward(self, obs):
        pass

class RecurrentACModel(BaseACModel):
    recurrent = True
    num_levels = 1

    @abstractmethod
    def forward(self, obs, memory):
        pass

    @property
    @abstractmethod
    def memory_size(self):
        pass
