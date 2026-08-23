import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import rl.torch_ac as torch_ac

lang_model = "gru"
instr_dim = 128

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


# Inspired by FiLMedBlock from https://arxiv.org/abs/1709.07871
class ExpertControllerFiLM(nn.Module):
    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=imm_channels, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(imm_channels)
        self.conv2 = nn.Conv2d(in_channels=imm_channels, out_channels=out_features, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)

        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.apply(init_params)

    def forward(self, x, y):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        out = x * self.weight(y).unsqueeze(2).unsqueeze(3) + self.bias(y).unsqueeze(2).unsqueeze(3)
        out = self.bn2(out)
        out = F.relu(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class CNN2(nn.Module):
    def __init__(self, in_channels=16, output_dim=128):
        super(CNN2, self).__init__()

        self.network = nn.Sequential(
            # Feature Extraction Part
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),

            # Head Part
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        # The forward pass is now just a single call
        return self.network(x)

class CNN3(nn.Module):
    def __init__(self, in_channels=16, output_dim=128):
        super(CNN3, self).__init__()

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

            # Head Part (untouched)
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.network(x)

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # zero-init: identity at load time

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, N, C/8)
        k = self.key(x).view(B, -1, H * W)                      # (B, C/8, N)
        attn = F.softmax(q @ k, dim=-1)                          # (B, N, N)
        v = self.value(x).view(B, -1, H * W)                    # (B, C, N)
        out = (v @ attn.permute(0, 2, 1)).view(B, C, H, W)
        return x + self.gamma * out


class MemoryCrossAttention(nn.Module):
    """Each spatial position scores its relevance against memory, then
    reweights the value features at that position accordingly.
    Answers: WHERE in the local view matters given what I remember."""
    def __init__(self, mem_dim, spatial_channels, embed_dim=64):
        super().__init__()
        self.q_space = nn.Conv2d(spatial_channels, embed_dim, 1)  # spatial queries
        self.k_mem   = nn.Linear(mem_dim, embed_dim)               # memory key
        self.v       = nn.Conv2d(spatial_channels, spatial_channels, 1)
        self.scale   = embed_dim ** -0.5
        self.gamma   = nn.Parameter(torch.zeros(1))  # zero-init: identity at load time

    def forward(self, h, x):
        B, C, H, W = x.shape
        N = H * W
        q = self.q_space(x).view(B, -1, N).permute(0, 2, 1)  # (B, N, E)
        k = self.k_mem(h).unsqueeze(2)                          # (B, E, 1)
        v = self.v(x).view(B, C, N)                            # (B, C, N)
        attn = F.softmax(q @ k * self.scale, dim=1)            # (B, N, 1)
        out = (v * attn.permute(0, 2, 1)).view(B, C, H, W)    # (B, C, H, W)
        return x + self.gamma * out


class FiLM(nn.Module):
    """Residual FiLM conditioned on memory. gamma=0 start — identity at init."""
    def __init__(self, mem_dim, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)
        self.scale = nn.Linear(mem_dim, channels)
        self.shift = nn.Linear(mem_dim, channels)
        self.gamma = nn.Parameter(torch.zeros(1))  # zero-init: identity at load time

    def forward(self, x, h):
        # FiLM+ResNet pattern: transform features before modulation so memory
        # scales/shifts a richer intermediate representation rather than the raw
        # input directly. Bare FiLM (no convs) modulates only what is already
        # there; the conv pair gives memory something more expressive to work with.
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        g = self.scale(h).unsqueeze(2).unsqueeze(3)
        b = self.shift(h).unsqueeze(2).unsqueeze(3)
        out = F.relu(self.bn2(out * g + b))
        return x + self.gamma * out


class CNN32Expert(nn.Module):
    """CNN32 extended with SelfAttention, MemoryCrossAttention and FiLM.

    The base CNN32 processes every frame independently: spatial positions
    never communicate with each other or with episodic memory.  This works
    for local obstacle avoidance but fails when the agent must integrate
    what it has already seen into where it should look next.

    Three complementary blocks are inserted at the 11x11 mid-resolution
    feature map (N=121 positions), after the first stride-2 downsample of
    the 21x21 input view.  At 21x21 the self-attention map is 441x441
    (expensive); after all downsamples it collapses to 3x3 (too few
    positions to be meaningful).  11x11 is the sweet spot.

    SelfAttention
        Each of the 121 positions attends to all others via a full N x N
        map.  Relates spatially distant features: a corridor entrance on
        one edge can influence how a door on the opposite edge is
        processed.  Solves navigation context -- room shapes, passage
        widths, geometric relationships between obstacles.

    MemoryCrossAttention
        Spatial positions are queries; the LSTM hidden state (episodic
        memory) is a single key.  Each position scores its relevance to
        what the agent currently remembers and reweights its own value
        features accordingly.  Answers WHERE in the local view matters
        given what was seen before: explored corridors get suppressed,
        unvisited areas and tracked entities get amplified.

    FiLM (Feature-wise Linear Modulation)
        A residual two-conv block conditioned on LSTM memory via learned
        per-channel scale and shift.  While CrossAttention selects where
        to look, FiLM modulates WHAT features to look for: in combat the
        memory can amplify monster-related channels; during exploration it
        can amplify barrel and door channels.

    All three blocks initialize their residual gate (gamma) to zero,
    making them exact mathematical identities at load time.  A pretrained
    CNN32 checkpoint produces bit-identical outputs on the first forward
    pass; the new blocks grow their contribution gradually as training
    continues, so no restart from random weights is needed."""

    def __init__(self, in_channels=16, output_dim=512, mem_dim=512):
        super().__init__()

        # Layers 0-8 of CNN32: 21x21 -> 11x11 at 128ch
        self.trunk_a = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Attention blocks operate at 128ch / 11x11 (N=121)
        self.self_attn  = SelfAttention(128)
        self.cross_attn = MemoryCrossAttention(mem_dim, 128)
        self.film       = FiLM(mem_dim, 128)

        # Layers 9-23 of CNN32: 11x11 -> 512-dim embedding
        self.trunk_b = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, output_dim),
        )

    def forward(self, x, h_prev=None):
        x = self.trunk_a(x)
        x = self.self_attn(x)
        if h_prev is not None:
            x = self.cross_attn(h_prev, x)
            x = self.film(x, h_prev)
        x = self.trunk_b(x)
        return x

    def load_from_cnn32(self, cnn32_conv_state):
        """Copy CNN32 image_conv weights into trunk_a and trunk_b.
        Pass the sub-dict scoped to image_conv (keys like 'network.0.weight')."""
        own = self.state_dict()
        suffixes = ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']

        for new_i in range(9):          # trunk_a covers CNN32 network.0..8
            for s in suffixes:
                src = f'network.{new_i}.{s}'
                dst = f'trunk_a.{new_i}.{s}'
                if src in cnn32_conv_state and dst in own:
                    own[dst] = cnn32_conv_state[src]

        for offset in range(15):        # trunk_b covers CNN32 network.9..23
            for s in suffixes:
                src = f'network.{9 + offset}.{s}'
                dst = f'trunk_b.{offset}.{s}'
                if src in cnn32_conv_state and dst in own:
                    own[dst] = cnn32_conv_state[src]

        self.load_state_dict(own)


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

class CNN35(nn.Module):
    def __init__(self, in_channels=16, output_dim=512):
        super(CNN35, self).__init__()

        self.network = nn.Sequential(
            # Initial conv
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # Block 1: downsample (64->128)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # Block 2: downsample (128->256)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # Block 3: downsample (256->512)
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # Pool to fixed size and flatten
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.network(x)

class CNN4(nn.Module):
    def __init__(self, in_channels=16, output_dim=2048):
        super(CNN4, self).__init__()

        self.network = nn.Sequential(
            # Initial convolution
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 1 (64 -> 128 channels)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 2 (128 -> 256 channels)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 3 (256 -> 512 channels) - This block is widened
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Head Part - Now an MLP Head
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1024), # Intermediate layer
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), # Regularization
            nn.Linear(1024, output_dim) # Final layer
        )

    def forward(self, x):
        return self.network(x)

# self.memory_rnn = nn.LSTMCell(...) stores temporal memory across steps;
# it takes current input features and previous hidden/cell states as input,
# and outputs a new hidden state (used as 'embedding') and updated memory.

# The memory tensor is updated per timestep using:
#   hidden = self.memory_rnn(x, (h_prev, c_prev))
#   memory = torch.cat([h, c], dim=1)  # for storage and future use

# self.word_embedding = nn.Embedding(obs_space["instr"], self.instr_dim)
# maps discrete instruction tokens into continuous vectors.

# self.instr_rnn = nn.GRU(...) encodes the sequence of embedded instruction tokens
# into outputs and final states, preserving the token order and linguistic context.

# If self.lang_model == 'attgru', attention is applied to instruction embeddings.
# The attention is computed via:
#   self.memory2key = nn.Linear(self.semi_memory_size, self.instr_dim)
# which projects the memory (hidden state) into a key vector space.

# The attention weight per word is computed using:
#   pre_softmax = (keys[:, None, :] * instr_embedding).sum(2) + 1000 * mask
# where keys come from self.memory2key(memory), and instr_embedding is the GRU output.

# The attention weight vector is applied to the instruction:
#   instr_embedding = (instr_embedding * attention[:, :, None]).sum(1)
# to produce a focused instruction embedding based on current memory.

# When the episode ends (obs.done is True), memory is reset to zero
# to prevent information leakage across episodes:
#   memory[obs.done] *= .0
#   see the analyze_feedback() function

# The memory state evolves during the episode and reflects task
# progress.  The memory is updated at each step, usually via an RNN
# (like a GRU or LSTM), which summarizes the agent's perception
# history. For example:
#   - Whether it's seen or picked up the red ball
#   - What objects or rooms it's visited
#   - Whether it's completed part of the task
# So memory serves as a temporal trace of the agent's experience so far.

# The attention mechanism over the instruction allows the agent to
# dynamically select which part of the instruction is relevant right
# now, based on its current memory state.

# keys = keys2memory(memory) is used to focus attention.
# keys2memory is a learned neural net (usually a linear or MLP layer).
# It transforms the current memory into a "query" vector.  This query
# is dot-multiplied with the embedded instruction words.  The softmax
# of those scores gives attention weights, which effectively say:
# "Given what I remember (memory), which word(s) in the instruction
# should I care about now?"

# This attention-weighted instruction is then used in FiLM modulation layers
# to adjust visual processing in self.controllers — e.g., for goal-relevant features.

# FiLM (Feature-wise Linear Modulation) is a technique to modulate
# neural activations based on another input — often used in
# multi-modal networks (e.g., language + vision). FiLM allows language
# (the instruction) to control how vision is processed — gating or
# enhancing visual features depending on what's said.

# What is "vector modulates the image"?
#
# Let's say the image shows a room, and the instruction is: “Go to the red key.”
# - The model encodes the instruction to a vector y.
# - That vector is transformed to gamma and beta.
# - The FiLM block uses these to enhance the channels that represent
#   red objects (maybe channel 42 is "red pixels"), and suppress
#   irrelevant features like walls or blue objects.
#
# So when we say "modulates the image", we mean:
# - The instruction vector controls which visual features are
#   emphasized or suppressed by changing the activation values in the
#   image tensor, channel-wise.

# All components (self.memory_rnn, self.instr_rnn, self.memory2key, attention logic,
# controllers, etc.) are trained end-to-end via PPO or another RL algorithm.
# https://medium.com/@dlgkswn3124/summary-squeeze-and-excitation-networks-senet-a510e902e668
#
class AutomapCNN(nn.Module):
    """Small CNN for the 40x40 automap observation (3 input channels).

    Channels: ch0=explored, ch1=frontier, ch2=player position.
    Three stride-2 convolutions collapse 40x40 -> 5x5, then AdaptiveAvgPool
    and a linear projection give a 64-dim embedding added as a residual to
    the LSTM output.  All layers use default PyTorch init, then FlatACModel
    normalizes them via apply(init_params) - see init_params() and
    FlatACModel.__init__ for the effective initial weight scale."""

    def __init__(self, output_dim=64):
        super().__init__()
        self.embed_dim = output_dim
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),   # 40->20
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 20->10
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 10->5
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, output_dim),
        )
    def forward(self, x):
        return self.network(x)


class FlatACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space,
                 cnn_arch,
                 embedding_dim=256,
                 use_memory=False,
                 use_text=False):
        super().__init__()

        # No hierarchy
        self.num_hierarchy_levels = 1

        # Decide which components are enabled
        self.cnn_arch = cnn_arch
        self.use_text = use_text
        self.use_memory = use_memory

        image_shape = obs_space["image"]
        in_channels = image_shape[-1]

        # Number of env-bit channels at the start of the image tensor.
        # Used by load_from_status to preserve env-bit conv filters when
        # padding a v1 checkpoint up to a v2 first-conv input width.
        self.nr_env_channels = obs_space.get("nr_env_channels")

        # Optional scalar branch (Diablo v2 obs). A small affine lift +
        # ReLU brings the [0,1] scalar features onto a scale comparable
        # to post-conv activations so the LSTM input weights see
        # balanced gradients across image and scalar columns.
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

        if self.cnn_arch == "cnn1":
            self.image_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2)),
                nn.ReLU()
            )
        elif self.cnn_arch == "cnn2":
            self.image_conv = CNN2(in_channels=in_channels, output_dim=embedding_dim)

        elif self.cnn_arch in ("cnn3", "cnn31"):
            self.image_conv = CNN3(in_channels=in_channels, output_dim=embedding_dim)

        elif self.cnn_arch == "cnn32":
            self.image_conv = CNN32(in_channels=in_channels, output_dim=embedding_dim)

        elif self.cnn_arch == "cnn32expert":
            self.image_conv = CNN32Expert(in_channels=in_channels,
                                          output_dim=embedding_dim,
                                          mem_dim=embedding_dim)

        elif self.cnn_arch == "cnn35":
            self.image_conv = CNN35(in_channels=in_channels, output_dim=embedding_dim)

        elif self.cnn_arch == "cnn4":
            self.image_conv = CNN4(in_channels=in_channels, output_dim=embedding_dim)

        elif self.cnn_arch.startswith("expert_filmcnn"):
            if not self.use_text:
                raise ValueError("FiLM architecture can be used when instructions are enabled")

            self.image_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=(2, 2), padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2)
            )
            self.film_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        else:
            raise ValueError("Incorrect architecture name: {}".format(self.cnn_arch))

        # Calculate image embedding size
        dummy = torch.zeros(1, *image_shape[::-1])
        if self.cnn_arch.startswith("expert_filmcnn"):
            dummy = self.image_conv(dummy)
            dummy = self.film_pool(dummy)
        else:
            dummy = self.image_conv(dummy)
        self.image_embedding_size = dummy.numel()

        # automap branch: 64-dim embedding fed into the LSTM input alongside
        # the env CNN and scalars so gradients reach the CNN every step.
        self.has_automap = "automap" in obs_space
        if self.has_automap:
            self.automap_enc = AutomapCNN(output_dim=64)
            self.automap_embed_size = self.automap_enc.embed_dim
        else:
            self.automap_embed_size = 0

        # Define memory. Scalars and automap (when present) are concatenated
        # with the image embedding at the LSTM input so the recurrence sees
        # spatial, global, and map state every step.
        if self.use_memory:
            rnn_input_size = (self.image_embedding_size + self.scalar_embed_size
                              + self.automap_embed_size)
            self.memory_rnn = nn.LSTMCell(rnn_input_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            #self.word_embedding_size = 128
            #self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            #self.text_embedding_size = 128
            #self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

            self.word_embedding = nn.Embedding(obs_space["text"], instr_dim)
            gru_dim = instr_dim
            if lang_model in ['bigru', 'attgru']:
                gru_dim //= 2
            self.text_rnn = nn.GRU(
                instr_dim, gru_dim, batch_first=True,
                bidirectional=(lang_model in ['bigru', 'attgru']))
            self.final_instr_dim = instr_dim

            if lang_model == 'attgru':
                self.memory2key = nn.Linear(self.memory_size, self.final_instr_dim)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        #if self.use_text:
        #    self.embedding_size += self.text_embedding_size
        if self.use_text and not "filmcnn" in self.cnn_arch:
            self.embedding_size += self.final_instr_dim
        # Without an RNN, scalars/automap cannot ride the recurrence; concat
        # them straight into the actor/critic input instead.
        if self.has_scalars and not self.use_memory:
            self.embedding_size += self.scalar_embed_size
        if self.has_automap and not self.use_memory:
            self.embedding_size += self.automap_embed_size

        if self.cnn_arch.startswith("expert_filmcnn"):
            if self.cnn_arch == "expert_filmcnn":
                num_module = 2
            else:
                num_module = int(self.cnn_arch[(self.cnn_arch.rfind('_') + 1):])
            self.controllers = []
            for ni in range(num_module):
                if ni < num_module-1:
                    mod = ExpertControllerFiLM(
                        in_features=self.final_instr_dim,
                        out_features=128, in_channels=128, imm_channels=128)
                else:
                    mod = ExpertControllerFiLM(
                        #in_features=self.final_instr_dim, out_features=self.image_dim,
                        in_features=self.final_instr_dim, out_features=self.image_embedding_size,
                        in_channels=128, imm_channels=128)
                self.controllers.append(mod)
                self.add_module('FiLM_Controler_' + str(ni), mod)

        if self.cnn_arch == "cnn4":
            # Define actor's model, which gradually reduces the feature
            # size for large embeddings, like:
            # 2048 -> 1024 -> 512 -> 256 -> action_space
            self.actor = nn.Sequential(
                nn.Linear(self.embedding_size, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, action_space.n)
            )

            # Define critic's model, has the same funnel structure as the
            # actor, but outputs a single value
            self.critic = nn.Sequential(
                nn.Linear(self.embedding_size, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
        elif self.cnn_arch in ("cnn32", "cnn32expert", "cnn35"):
            # Define actor's model, which gradually reduces the feature
            # size for large embeddings, like:
            # 512 -> 256 -> action_space
            self.actor = nn.Sequential(
                nn.Linear(self.embedding_size, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, action_space.n)
            )

            # Define critic's model, has the same funnel structure as the
            # actor, but outputs a single value
            self.critic = nn.Sequential(
                nn.Linear(self.embedding_size, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        elif self.cnn_arch == "cnn31":
            # Define actor's model
            self.actor = nn.Sequential(
                nn.Linear(self.embedding_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_space.n)
            )

            # Define critic's model
            self.critic = nn.Sequential(
                nn.Linear(self.embedding_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        else:
            # Define actor's model
            self.actor = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, action_space.n)
            )

            # Define critic's model
            self.critic = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )

        # Initialize parameters correctly
        self.apply(init_params)

        # Re-init automap final linear with small sigma AFTER apply(init_params)
        # which would otherwise overwrite it with unit-norm rows (rms~0.125).
    def load_from_status(self, status, logger=None):
        if (self.cnn_arch == "cnn32expert" and
                "image_conv.network.0.weight" in status["model_state"]):
            if logger:
                logger.info("Converted cnn32 -> cnn32expert weights\n")
            status.pop("optimizer_state", None)
            self.load_from_cnn32_status(status)
            return

        src = status["model_state"]
        own = self.state_dict()
        has_mismatch = any(k in own and src[k].shape != own[k].shape
                           for k in src)
        missing = [k for k in own if k not in src]
        if has_mismatch:
            if logger:
                logger.info("Detected obs/action shape mismatch, "
                            "applying zero-pad surgery\n")
            status.pop("optimizer_state", None)
            self.load_state_dict(self._pad_state_dict(src))
        elif missing:
            if logger:
                logger.info("Checkpoint missing keys (new branch): %s\n"
                            % ", ".join(missing))
            status.pop("optimizer_state", None)
            self.load_state_dict(src, strict=False)
        else:
            self.load_state_dict(src)

    def _pad_state_dict(self, src):
        """Build a state_dict for self by copying matching tensors from src
        and zero-padding mismatched ones to self's shapes.

        Two pad strategies:
          - First-conv weight (4D, matching out_channels and kernel size,
            differing in_channels): preserve only the env-bit channel
            prefix. v1's env-status broadcast filters and v2's new
            monster_attrs channels both end up zero -- the env-status
            filters were trained for constant-valued planes and would
            misbehave on the per-tile floats that occupy those indices
            in v2.
          - All other mismatched tensors: left-align src in the new
            shape, zero-fill the trailing positions. Covers the LSTM
            input weight (new scalar columns at the right edge of W_ih)
            and the actor's last linear weight/bias (new action rows at
            the bottom).

        Keys in src that aren't in own state_dict are dropped. Keys in
        own that aren't in src keep their freshly-initialized values
        (e.g. the new scalars_enc module on a v1 -> v2 upgrade).
        """
        own = self.state_dict()
        for k, v in src.items():
            if k not in own:
                continue
            dst = own[k]
            if v.shape == dst.shape:
                own[k] = v
                continue
            new = torch.zeros(dst.shape, dtype=v.dtype, device=v.device)
            is_first_conv = (v.dim() == 4 and
                             v.shape[0] == dst.shape[0] and
                             tuple(v.shape[2:]) == tuple(dst.shape[2:]) and
                             v.shape[1] != dst.shape[1] and
                             self.nr_env_channels is not None)
            if is_first_conv:
                n = min(self.nr_env_channels, v.shape[1], dst.shape[1])
                new[:, :n, ...] = v[:, :n, ...]
            else:
                slices = tuple(slice(0, min(s, d))
                               for s, d in zip(v.shape, dst.shape))
                new[slices] = v[slices]
            own[k] = new
        return own

    def load_from_cnn32_status(self, status):
        """Bootstrap a cnn32expert model from a pretrained cnn32 status.
        Trunk weights are remapped; all new blocks stay at gamma=0 (identity)."""
        assert self.cnn_arch == "cnn32expert", "Only valid for cnn32expert"
        src = status["model_state"]

        # Remap image_conv.network.* -> trunk_a/trunk_b
        conv_state = {k[len("image_conv."):]: v
                      for k, v in src.items() if k.startswith("image_conv.")}
        self.image_conv.load_from_cnn32(conv_state)

        # Copy everything else (actor, critic, memory_rnn) directly
        own = self.state_dict()
        for k, v in src.items():
            if not k.startswith("image_conv.") and k in own:
                own[k] = v
        self.load_state_dict(own)

    def save_to_status(self, status):
        status.update({"model_state": self.state_dict(),
                       "num_hierarchy_levels": self.num_hierarchy_levels,
                       "model_class": type(self).__name__})

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory, **kwargs):
        if self.use_text:
            embed_text = self._get_embed_text(obs.text)

            if lang_model == "attgru":
                # outputs: B x L x D
                # memory: B x M
                mask = (obs.instr != 0).float()
                embed_text = embed_text[:, :mask.shape[1]]
                # If memory is zeroed out (episone is done) keys will
                # be near-zero if self.memory2key is a Linear layer
                # with no bias
                keys = self.memory2key(memory)
                # When keys are near-zero (memory is zeroed out)
                # pre_softmax becomes almost uniform across non-zero
                # tokens (thanks to `+ 1000 * mask`)
                pre_softmax = (keys[:, None, :] * embed_text).sum(2) + 1000 * mask
                attention = F.softmax(pre_softmax, dim=1)
                # When memory is meaningful (not zero), attention is
                # sharper and selects the most relevant tokens for the
                # current state.
                embed_text = (embed_text * attention[:, :, None]).sum(1)

        # Convert (B, H, W, C) to (B, C, H, W)
        x = obs.image.transpose(1, 3).transpose(2, 3)

        if self.cnn_arch.startswith("expert_filmcnn"):
            x = self.image_conv(x)
            for controler in self.controllers:
                x = controler(x, embed_text)
            x = F.relu(self.film_pool(x))
        elif self.cnn_arch == "cnn32expert":
            h_prev = memory[:, :self.semi_memory_size]
            x = self.image_conv(x, h_prev)
        else:
            x = self.image_conv(x)

        # Flatten (B, C, H, W) to (B, C x H x W)
        x = x.reshape(x.shape[0], -1)

        if self.has_scalars:
            s = self.scalars_enc(obs.scalars)

        if self.has_automap:
            am = obs.automap.transpose(1, 3).transpose(2, 3)  # (B,H,W,C) -> (B,C,H,W)
            a = self.automap_enc(am)

        if self.use_memory:
            rnn_in = torch.cat(
                [x] + ([s] if self.has_scalars else [])
                + ([a] if self.has_automap else []),
                dim=1)
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(rnn_in, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = torch.cat(
                [x] + ([s] if self.has_scalars else [])
                + ([a] if self.has_automap else []),
                dim=1)

        if self.use_text and not "filmcnn" in self.cnn_arch:
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=x)

        x = self.critic(embedding)
        value = x

        # Distribution is a list of categorical objects for each level
        # (L is 1 for this model), and the value is expected to be in
        # the (P, L) shape.
        return [dist], value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
