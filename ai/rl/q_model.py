"""FlatQModel: CNN + LSTM + Dueling Q-head for pure Q-learning.

Same backbone as FlatACModel (CNN, optional scalars, optional automap, LSTM)
but replaces actor/critic with a single DuelingQHead.

forward(obs, memory) -> (q_all, memory)
  q_all : (B, n_actions, N) - distributional Q over quantiles
  memory: (B, memory_size)  - updated LSTM state [h, c]
"""

import torch
import torch.nn as nn

from torch.distributions.categorical import Categorical

from rl.torch_ac.model import RecurrentACModel
from rl.flat_model import (CNN32, CNN32Expert, CNN35, CNN3, CNN2, CNN4,
                            AutomapCNN, init_params)
from rl.q_critic import DuelingQHead


class FlatQModel(nn.Module, RecurrentACModel):
    """CNN + LSTM + Dueling Q-head for QR-DQN Q-learning."""

    num_hierarchy_levels = 1

    def __init__(self, obs_space, action_space,
                 cnn_arch="cnn32expert",
                 embedding_dim=256,
                 num_quantiles=32):
        super().__init__()

        self.cnn_arch = cnn_arch
        image_shape = obs_space["image"]
        in_channels = image_shape[-1]
        self.nr_env_channels = obs_space.get("nr_env_channels")

        self.has_scalars = "scalars" in obs_space
        if self.has_scalars:
            scalar_dim = obs_space["scalars"][0]
            self.scalar_embed_size = 64
            self.scalars_enc = nn.Sequential(
                nn.Linear(scalar_dim, self.scalar_embed_size),
                nn.ReLU(),
            )
        else:
            self.scalar_embed_size = 0

        if cnn_arch == "cnn32":
            self.image_conv = CNN32(in_channels=in_channels, output_dim=embedding_dim)
        elif cnn_arch == "cnn32expert":
            self.image_conv = CNN32Expert(in_channels=in_channels,
                                          output_dim=embedding_dim,
                                          mem_dim=embedding_dim)
        elif cnn_arch == "cnn35":
            self.image_conv = CNN35(in_channels=in_channels, output_dim=embedding_dim)
        elif cnn_arch in ("cnn3", "cnn31"):
            self.image_conv = CNN3(in_channels=in_channels, output_dim=embedding_dim)
        elif cnn_arch == "cnn2":
            self.image_conv = CNN2(in_channels=in_channels, output_dim=embedding_dim)
        elif cnn_arch == "cnn4":
            self.image_conv = CNN4(in_channels=in_channels, output_dim=embedding_dim)
        else:
            raise ValueError(f"Unsupported cnn_arch for FlatQModel: {cnn_arch}")

        dummy = torch.zeros(1, *image_shape[::-1])
        self.image_embedding_size = self.image_conv(dummy).numel()

        self.has_automap = "automap" in obs_space
        if self.has_automap:
            self.automap_enc = AutomapCNN(output_dim=64)
            self.automap_embed_size = self.automap_enc.embed_dim
        else:
            self.automap_embed_size = 0

        rnn_input_size = (self.image_embedding_size + self.scalar_embed_size
                          + self.automap_embed_size)
        self.memory_rnn = nn.LSTMCell(rnn_input_size, self.semi_memory_size)

        self.apply(init_params)

        self.q_head = DuelingQHead(self.semi_memory_size, action_space.n, num_quantiles)

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory, **kwargs):
        x = obs.image.transpose(1, 3).transpose(2, 3)  # (B,H,W,C) -> (B,C,H,W)

        if self.cnn_arch == "cnn32expert":
            h_prev = memory[:, :self.semi_memory_size]
            x = self.image_conv(x, h_prev)
        else:
            x = self.image_conv(x)

        x = x.reshape(x.shape[0], -1)

        rnn_parts = [x]
        if self.has_scalars:
            rnn_parts.append(self.scalars_enc(obs.scalars))
        if self.has_automap:
            am = obs.automap.transpose(1, 3).transpose(2, 3)
            rnn_parts.append(self.automap_enc(am))

        rnn_in = torch.cat(rnn_parts, dim=1)
        hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
        hidden = self.memory_rnn(rnn_in, hidden)
        h = hidden[0]
        memory = torch.cat(hidden, dim=1)

        q_all = self.q_head(h)  # (B, A, N)
        return q_all, memory

    def save_to_status(self, status):
        status.update({"model_state": self.state_dict(),
                       "num_hierarchy_levels": self.num_hierarchy_levels,
                       "model_class": type(self).__name__})

    def load_from_status(self, status, logger=None):
        src = status.get("model_state", {})
        if not src:
            return
        own = self.state_dict()
        missing = [k for k in own if k not in src]
        if missing and logger:
            logger.info("FlatQModel: missing keys in checkpoint: %s\n" % ", ".join(missing))
        self.load_state_dict(src, strict=False)


class QModelEvalWrapper:
    """Wraps FlatQModel to match the (dist, value, memory) interface for batch_evaluate.

    Converts Q-values to a Categorical distribution so argmax eval works with
    the existing evaluation code without modification.
    argmax(Q_mean) == argmax(softmax(Q_mean)) so greedy eval is exact.
    """
    def __init__(self, q_model):
        self.q_model = q_model
        self.recurrent = q_model.recurrent
        self.memory_size = q_model.memory_size
        self.num_hierarchy_levels = q_model.num_hierarchy_levels

    def __call__(self, obs, memory, **kwargs):
        q_all, new_memory = self.q_model(obs, memory)
        dist = Categorical(logits=q_all.mean(-1))  # (B, A) -> greedy ~ argmax(Q_mean)
        return [dist], None, new_memory
