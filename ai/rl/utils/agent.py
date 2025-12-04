import torch
import numpy as np

import rl.utils as utils
from .other import device
from rl.model import ACModel


class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, acmodel, preprocess_obss, argmax, num_levels, num_envs):
        self.acmodel = acmodel
        self.preprocess_obss = preprocess_obss
        self.argmax = argmax
        self.num_levels = num_levels
        self.num_envs = num_envs
        if self.acmodel.recurrent:
            self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size, device=device)

    @classmethod
    def from_internal_model(agent_cls, obs_space, action_space,
                            model_dir, cnn_arch, best=False,
                            argmax=False, num_envs=1, embedding_dim=256,
                            use_memory=False, use_text=False):
        obs_space, preprocess_obss = utils.get_obss_preprocessor(obs_space)
        acmodel = ACModel(obs_space, action_space, cnn_arch,
                          embedding_dim=embedding_dim,
                          use_memory=use_memory,
                          use_text=use_text)
        acmodel.load_state_dict(utils.get_model_state(model_dir, best=best))
        acmodel.to(device)
        acmodel.eval()
        if hasattr(preprocess_obss, "vocab"):
            preprocess_obss.vocab.load_vocab(utils.get_vocab(model_dir))

        return agent_cls(acmodel, preprocess_obss, argmax, num_envs)

    @classmethod
    def from_external_model(agent_cls, acmodel, obs_space,
                            argmax=False, num_envs=1):
        _, preprocess_obss = utils.get_obss_preprocessor(obs_space)

        return agent_cls(acmodel, preprocess_obss, argmax, num_envs)

    @staticmethod
    def get_active_indices(size, active_indices):
        if active_indices is None:
            active_indices = np.arange(size)
        else:
            assert size == len(active_indices)
        return active_indices

    def get_actions(self, obss, active_indices=None):
        active_indices = Agent.get_active_indices(len(obss), active_indices)

        with torch.no_grad():
            preprocessed_obss = self.preprocess_obss(obss, device=device)
            if self.acmodel.recurrent:
                memories = self.memories[active_indices]
                dist, _, memories = self.acmodel(preprocessed_obss, memories)
                self.memories[active_indices] = memories
            else:
                dist, _ = self.acmodel(preprocessed_obss)

        assert len(dist) == self.num_levels

        # Actions shape (P, L)
        if self.argmax:
            actions = torch.stack([d.probs.argmax(dim=1) for d in dist], dim=1)
        else:
            actions = torch.stack([d.sample() for d in dist], dim=1)

        return actions.cpu().numpy()

    def get_action(self, obs):
        return self.get_actions([obs])[0]

    def analyze_feedbacks(self, rewards, dones, active_indices=None):
        assert len(rewards) == len(dones)
        active_indices = Agent.get_active_indices(len(dones), active_indices)

        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float, device=device).unsqueeze(1)
            self.memories[active_indices] *= masks

    def analyze_feedback(self, reward, done):
        assert np.isscalar(reward) and np.isscalar(done)
        return self.analyze_feedbacks([reward], [done])
