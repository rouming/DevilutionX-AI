"""
Hierarchy RL model with 1 manager, 1 worker and 2 options: explorer and combat

Author: Roman Penyaev <r.peniaev@gmail.com>
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from rl.torch_ac.utils.sampling import deterministic_sample
import rl.torch_ac as torch_ac

def log_info(logger, text):
    if logger:
        logger.info(text)
    else:
        print(text)

class CNN32(nn.Module):
    def __init__(self, in_channels=16, output_dim=512):
        super(CNN32, self).__init__()

        self.network = nn.Sequential(
            # Initial convolution
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # This layer doubles the channels (64->128) and halves the grid size (stride=2)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # Downsamples
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # No downsampling
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # This layer doubles the channels (128->256) and halves the grid size (stride=2)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # Downsamples
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # No downsampling
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # This layer doubles the channels (256->512) and halves the grid size (stride=2)
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), # Downsamples
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # No downsampling
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Head Part (untouched)
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.network(x)

MANAGER_LEVEL = 1

class HRLACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, cnn_arch='cnn32',
                 embedding_dim=512, num_options=2,
                 use_memory=True, use_text=False):
        super().__init__()

        assert use_memory
        assert not use_text

        # Classic Management and Workers layers
        self.num_hierarchy_levels = 2

        # Decide which components are enabled
        self.cnn_arch = cnn_arch
        self.use_text = False
        self.use_memory = True

        image_shape = obs_space["image"]
        in_channels = image_shape[-1]

        self.image_conv = CNN32(in_channels=in_channels, output_dim=embedding_dim)

        # Calculate image embedding size
        dummy = torch.zeros(1, *image_shape[::-1])
        self.image_embedding_size = self.image_conv(dummy).numel()
        self.embedding_size = self.semi_memory_size

        # Define memory
        self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        #
        # Manager, level 1
        #
        self.manager_actor = nn.Sequential(
            nn.Linear(self.embedding_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_options)
        )
        self.manager_critic = nn.Sequential(
            nn.Linear(self.embedding_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # V(s)
        )

        #
        # Worker, level 2
        #
        # Options to a dense vector
        self.option_emb_dim = 64
        self.option_embedding = nn.Embedding(num_options, self.option_emb_dim)

        worker_input_dim = self.embedding_size + self.option_emb_dim

        self.worker_actor = nn.Sequential(
            nn.Linear(worker_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_space.n) # Atomic actions
        )
        self.worker_critic = nn.Sequential(
            nn.Linear(worker_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Initialize parameters
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory, *, noise=None, action=None):
        # Convert (B, H, W, C) to (B, C, H, W)
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        # Flatten (B, C, H, W) to (B, C x H x W)
        x = x.reshape(x.shape[0], -1)

        hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
        hidden = self.memory_rnn(x, hidden)
        embedding = hidden[0]
        new_memory = torch.cat(hidden, dim=1)

        # Manager pass
        manager_logits = self.manager_actor(embedding)
        dist_manager = Categorical(logits=manager_logits)

        if action is not None:
            # TRAINING MODE - we must use the option that was actually
            # taken in the history to correctly train the worker.
            assert self.training and noise is None
            active_option = action[:, MANAGER_LEVEL]
        else:
            # COLLECTION MODE - we use stateless and deterministic
            # categorical sampling
            assert noise is not None
            active_option = deterministic_sample(dist_manager.probs, noise[:, MANAGER_LEVEL])

        # Worker pass, embed the chosen option
        opt_emb = self.option_embedding(active_option) # (B, 64)

        # Concatenate: visual + option
        worker_input = torch.cat([embedding, opt_emb], dim=1)

        worker_logits = self.worker_actor(worker_input)
        dist_worker = Categorical(logits=worker_logits)

        # Value estimation
        val_manager = self.manager_critic(embedding)
        val_worker = self.worker_critic(worker_input)

        # Combine values into shape (P, L)
        values = torch.cat([val_worker, val_manager], dim=1)

        # Return list of distributions: [worker, manager]
        return [dist_worker, dist_manager], values, new_memory

    def load_from_status(self, status, logger=None):
        if status.get("num_hierarchy_levels", 1) == 2:
            self.load_state_dict(status["model_state"])
        else:
            # In the case of conversion, the old optimizer state is ignored.
            status.pop("optimizer_state", None)
            self.load_flat_model(status["model_state"], logger)

    def save_to_status(self, status):
        status.update({"model_state": self.state_dict(),
                       "num_hierarchy_levels": self.num_hierarchy_levels})

    def load_flat_model(self, old_state, logger):
        """Loads weights from a flat PPO model into the encoder,
        memory and worker Performs surgery on the first linear layer
        to accommodate the new Option Embedding.
        """
        # Get current model state
        new_state = self.state_dict()

        log_info(logger, "Trying to load a flat model into the HRL model")

        # Iterate and Copy
        for name, new_param in new_state.items():
            # Direct copy for those which match exactly,
            # e.g. "image_conv...", "memory_rnn..."
            if name in old_state and old_state[name].shape == new_param.shape:
                new_state[name].copy_(old_state[name])
                log_info(logger, f"Loaded directly: {name}")
                continue

            # Weight surgery for actor and critic
            # Map old "actor.0.weight" -> new "worker_actor.0.weight"
            old_name = name.replace("worker_", "")

            if old_name in old_state:
                old_param = old_state[old_name]

                # Check if this is the specific layer that needs surgery
                # (e.g. the first linear layer connected to input)
                if new_param.shape != old_param.shape:
                    log_info(logger, f"Splicing (surgery): {name} | new: {new_param.shape} <- old: {old_param.shape}")

                    # Assume dimension 1 is the input dimension (in_features)
                    # new_param shape: [Out_Features, Shared_Emb + Option_Emb]
                    # old_param shape: [Out_Features, Shared_Emb]

                    dim_shared = old_param.shape[1]

                    # Copy the old weights into the beginning
                    new_param[:, :dim_shared].copy_(old_param)

                    # Zero-init the new weights (option part)
                    # This ensures the option has NO effect at start
                    # (Preserves pre-trained behavior)
                    nn.init.constant_(new_param[:, dim_shared:], 0.0)

                else:
                    # Direct copy if dimensions match (e.g. layers 2, 3, output head)
                    new_param.copy_(old_param)
                    log_info(logger, f"Loaded directly: {name}")

        # Load the spliced state back into the model
        self.load_state_dict(new_state)
        log_info(logger, "Pre-trained worker loaded successfully with zero-initialized option inputs")
