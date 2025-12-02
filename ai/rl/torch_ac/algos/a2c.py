"""
HRL Adaptation of torch-ac,
based on torch-ac by lcswillems.

Changes:
- Adapted storage and collection to support 'num_levels' dimension (P x L).
- Manager and Worker steps are aligned for joint optimization.
- Implemented shared Encoder/Memory with multi-head outputs.
- Added 'opt_mask' to handle Truncated BPTT at option boundaries
  (detaching memory on option switch).

Author: Roman Penyaev, 2025
"""

import numpy
import torch
import torch.nn.functional as F

from rl.torch_ac.algos.base import BaseAlgo

class A2CAlgo(BaseAlgo):
    """The Advantage Actor-Critic algorithm."""

    def __init__(self, penv_pool, acmodel, device=None, num_levels=1, num_frames_per_proc=None, discount=0.99, lr=0.01, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-8, preprocess_obss=None, reshape_reward=None):
        num_frames_per_proc = num_frames_per_proc or 8

        super().__init__(penv_pool, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward)

        self.optimizer = torch.optim.RMSprop(self.acmodel.parameters(), lr,
                                             alpha=rmsprop_alpha, eps=rmsprop_eps)

    def update_parameters(self, exps, apply_update=True):
        # Compute starting indexes

        inds = self._get_starting_indexes()

        # Initialize update values

        update_entropy = np.zeros((self.num_levels, ))
        update_value = np.zeros((self.num_levels, ))
        update_policy_loss = np.zeros((self.num_levels, ))
        update_value_loss = np.zeros((self.num_levels, ))
        update_kl = np.zeros((self.num_levels, ))

        # Will be promoted to a tensor on the correct device
        update_loss_tensor = 0

        # Initialize memory

        if self.acmodel.recurrent:
            memory = exps.memory[inds]

        for i in range(self.recurrence):
            # Create a sub-batch of experience

            sb = exps[inds + i]

            # Compute loss

            if self.acmodel.recurrent:
                # We detach memory if and only if the option
                # changed, to prevent interference between
                # temporally extended skills.
                h = memory * sb.mask # episode resets
                h = h * sb.opt_mask + h.detach() * (1 - sb.opt_mask) # option boundaries
                dist, value, memory = self.acmodel(sb.obs, h)
            else:
                dist, value = self.acmodel(sb.obs)

            num_seqs = len(inds)

            assert len(dist) == self.num_levels
            assert value.shape == (num_seqs, self.num_levels)

            entropy = torch.stack([d.entropy().mean() for d in dist])

            new_log_prob = torch.stack([dist[j].log_prob(sb.actions[:, j])
                                        for j in range(self.num_levels)],
            policy_loss = -(new_log_prob * sb.advantage).mean(axis=0)

            value_loss = (value - sb.returnn).pow(2).mean(axis=0)

            loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

            # Kullback-Leibler (KL) divergence, not strictly needed
            # for A2C, but can be a good health check
            kl = (sb.log_prob - dist.log_prob(sb.action)).mean(axis=0)

            # Update batch values

            update_entropy += entropy.cpu().numpy()
            update_value += value.mean(axis=0).cpu().numpy()
            update_policy_loss += policy_loss.cpu().numpy()
            update_value_loss += value_loss.cpu().numpy()
            update_kl += kl.cpu().numpy()
            update_loss_tensor += loss

        # Update update values

        update_entropy /= self.recurrence
        update_value /= self.recurrence
        update_policy_loss /= self.recurrence
        update_value_loss /= self.recurrence
        update_loss_tensor /= self.recurrence
        update_kl /= self.recurrence

        # Update actor-critic

        if apply_update:
            self.optimizer.zero_grad()
            update_loss_tensor.sum().backward()
            update_grad_norm = torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(),
                                                              self.max_grad_norm).item()
            self.optimizer.step()
        else:
            update_grad_norm = 0.0

        # Log some values

        logs = {
            "entropy": update_entropy,
            "value": update_value,
            "policy_loss": update_policy_loss,
            "value_loss": update_value_loss,
            "kl": update_kl,
            "grad_norm": update_grad_norm,
        }

        return logs

    def _get_starting_indexes(self):
        """Gives the indexes of the observations given to the model and the
        experiences used to compute the loss at first.

        The indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`. If the model is not recurrent, they are all the
        integers from 0 to `self.num_frames`.

        Returns
        -------
        starting_indexes : list of int
            the indexes of the experiences to be used at first
        """

        starting_indexes = numpy.arange(0, self.num_frames, self.recurrence)
        return starting_indexes
