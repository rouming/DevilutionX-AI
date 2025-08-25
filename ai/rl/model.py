import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac

# XXX
arch = "cnn1"
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
# (like a GRU or LSTM), which summarizes the agent’s perception
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
# enhancing visual features depending on what’s said.

# What is "vector modulates the image"?
#
# Let’s say the image shows a room, and the instruction is: “Go to the red key.”
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

class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        image_shape = obs_space["image"]
        nr_channels = image_shape[-1]

        if arch == "cnn1":
            self.image_conv = nn.Sequential(
                nn.Conv2d(in_channels=nr_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),   # 21 -> 10
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),   # 10 -> 5
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))  # always -> 128-dim
            )
        elif arch.startswith("expert_filmcnn"):
            if not self.use_text:
                raise ValueError("FiLM architecture can be used when instructions are enabled")

            self.image_conv = nn.Sequential(
                nn.Conv2d(in_channels=nr_channels, out_channels=128, kernel_size=(2, 2), padding=1),
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
            raise ValueError("Incorrect architecture name: {}".format(arch))

        # Calculate image embedding size
        dummy = torch.zeros(1, *image_shape[::-1])
        if arch.startswith("expert_filmcnn"):
            dummy = self.image_conv(dummy)
            dummy = self.film_pool(dummy)
        else:
            dummy = self.image_conv(dummy)
        self.image_embedding_size = dummy.numel()

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

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
        if self.use_text and not "filmcnn" in arch:
            self.embedding_size += self.final_instr_dim

        if arch.startswith("expert_filmcnn"):
            if arch == "expert_filmcnn":
                num_module = 2
            else:
                num_module = int(arch[(arch.rfind('_') + 1):])
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

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        if self.use_text:
            embed_text = self._get_embed_text(obs.text)

            if lang_model == "attgru":
                # outputs: B x L x D
                # memory: B x M
                mask = (obs.instr != 0).float()
                embed_text = embed_text[:, :mask.shape[1]]
                # If memory is zeroed out (episone is done, see the
                # analyze_feedback()) keys will be near-zero if
                # self.memory2key is a Linear layer with no bias
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

        if arch.startswith("expert_filmcnn"):
            x = self.image_conv(x)
            for controler in self.controllers:
                x = controler(x, embed_text)
            x = F.relu(self.film_pool(x))
        else:
            x = self.image_conv(x)

        # Flatten (B, C, H, W) to (B, C x H x W)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text and not "filmcnn" in arch:
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
