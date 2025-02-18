import gymnasium as gym
import json
import math
import numpy
import os
import re
import rl.torch_ac as torch_ac
import torch

import rl.utils as utils


def get_obss_preprocessor(obs_space):
    # Check if it is a Diablo observation space
    if (isinstance(obs_space, gym.spaces.Dict) and
        {"env", "env-status"} <= set(obs_space.spaces)):

        env_space = obs_space.spaces["env"]
        env_status_space = obs_space.spaces["env-status"]
        nr_env_channels = int(math.log2(env_space.high.max() + 1))
        env_status_space_high = env_status_space.high.max()
        nr_channels = nr_env_channels + env_status_space.shape[-1]
        obs_space = { "image": (*env_space.shape, nr_channels) }

        def preprocess_obss(obss, device=None):
            env = numpy.array([obs["env"] for obs in obss])
            env_status = numpy.array([obs["env-status"] for obs in obss])
            return torch_ac.DictList({
                "image": batch_dungeon_observations_to_one_hot(env,
                                                               env_status,
                                                               nr_env_channels,
                                                               env_status_space_high,
                                                               device=device),
            })

    # Check if obs_space is an image space
    elif isinstance(obs_space, gym.spaces.Box):
        obs_space = {"image": obs_space.shape}

        def preprocess_obss(obss, device=None):
            return torch_ac.DictList({
                "image": preprocess_images(obss, device=device)
            })

    # Check if it is a MiniGrid observation space
    elif isinstance(obs_space, gym.spaces.Dict) and "image" in obs_space.spaces.keys():
        obs_space = {"image": obs_space.spaces["image"].shape, "text": 100}

        vocab = Vocabulary(obs_space["text"])

        def preprocess_obss(obss, device=None):
            return torch_ac.DictList({
                "image": preprocess_images([obs["image"] for obs in obss], device=device),
                "text": preprocess_texts([obs["mission"] for obs in obss], vocab, device=device)
            })

        preprocess_obss.vocab = vocab

    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space, preprocess_obss


def batch_dungeon_observations_to_one_hot(env, env_status,
                                          nr_env_channels,
                                          env_status_space_high,
                                          device=None):
    """
    Convert a batch of dungeon observations into a one-hot style tensor.

    Batch args:
      env_status (np.ndarray): 1D environment status vector
      env (np.ndarray): 2D dungeon map with bitfield-encoded flags

    Returns:
      torch.tensor: combined observation tensor
    """

    # (B, H, W)
    env_t = torch.as_tensor(env, dtype=torch.int64, device=device)
    # (B, S)
    env_status_t = torch.as_tensor(env_status, dtype=torch.float32, device=device)

    bit_indices = torch.arange(nr_env_channels, device=device, dtype=torch.int64)
    # (B, H, W, C)
    bit_planes = ((env_t.unsqueeze(-1) >> bit_indices) & 1).to(torch.float32)
    # (B, 1, 1, S)
    env_status_t = torch.clamp(env_status_t / env_status_space_high, 0.0, 1.0).unsqueeze(1).unsqueeze(2)
    # (B, H, W, S)
    env_status_t = env_status_t.expand(-1, env.shape[1], env.shape[2], -1)
    return torch.cat([bit_planes, env_status_t], dim=-1)


def preprocess_images(images, device=None):
    # Bug of Pytorch: very slow if not first converted to numpy array
    if isinstance(images, list):
        # Merges lists of ndarrays on the fast path
        images = numpy.array(images)
    elif isinstance(images, numpy.ndarray):
        if images.dtype == object:
            # Merges 1D object array in 4D array
            images = numpy.stack(images, axis=0)
        else:
            # Already perfect
            pass
    else:
        raise TypeError(f"Unsupported images type: {type(obss)}")

    return torch.tensor(images, device=device, dtype=torch.float)


def preprocess_texts(texts, vocab, device=None):
    var_indexed_texts = []
    max_text_len = 0

    for text in texts:
        tokens = re.findall("([a-z]+)", text.lower())
        var_indexed_text = numpy.array([vocab[token] for token in tokens])
        var_indexed_texts.append(var_indexed_text)
        max_text_len = max(len(var_indexed_text), max_text_len)

    indexed_texts = numpy.zeros((len(texts), max_text_len))

    for i, indexed_text in enumerate(var_indexed_texts):
        indexed_texts[i, :len(indexed_text)] = indexed_text

    return torch.tensor(indexed_texts, device=device, dtype=torch.long)


class Vocabulary:
    """A mapping from tokens to ids with a capacity of `max_size` words.
    It can be saved in a `vocab.json` file."""

    def __init__(self, max_size):
        self.max_size = max_size
        self.vocab = {}

    def load_vocab(self, vocab):
        self.vocab = vocab

    def __getitem__(self, token):
        if not token in self.vocab.keys():
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]
