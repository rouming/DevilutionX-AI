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

class PPOAlgo(BaseAlgo):
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, penv_pool, acmodel, device=None, num_levels=1, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(penv_pool, acmodel, device, num_levels, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        assert self.batch_size % self.recurrence == 0

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)
        self.batch_num = 0

    def update_parameters(self, exps, apply_update=True):
        # Collect experiences

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []
            log_kls = []

            for inds in self._get_batches_starting_indexes():
                # Initialize batch values

                batch_entropy = np.zeros((self.num_levels, ))
                batch_value = np.zeros((self.num_levels, ))
                batch_policy_loss = np.zeros((self.num_levels, ))
                batch_value_loss = np.zeros((self.num_levels, ))
                batch_kl = np.zeros((self.num_levels, ))

                # Will be promoted to a tensor on the correct device
                batch_loss_tensor = 0

                # If model is recurrent, load the stored hidden
                # states.  Mmemory state is detached from the
                # backpropagation (see the last few lines memory
                # related in the recurrence loop)
                if self.acmodel.recurrent:
                    # memory shape: (seqs_per_mb, hidden_size)
                    memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience

                    sb = exps[inds + i]

                    # sb.obs: tensor [S, ...]  (S = sequences_per_subbatch = batch_size // recurrence)
                    # sb.action, sb.log_prob, sb.advantage, sb.returnn, sb.value are all aligned [S, ...]

                    # Compute loss

                    if self.acmodel.recurrent:
                        # We detach memory if and only if the option
                        # changed, to prevent interference between
                        # temporally extended skills.
                        h = memory * sb.mask # episode resets
                        h = h * sb.opt_mask + h.detach() * (1 - sb.opt_mask) # option boundaries
                        # Recurrent chain through memory
                        dist, value, memory = self.acmodel(sb.obs, h)
                    else:
                        dist, value = self.acmodel(sb.obs)

                    num_seqs = len(inds)

                    assert len(dist) == self.num_levels
                    assert value.shape == (num_seqs, self.num_levels)

                    # Entropy (scalar) averaged over the sub-batch (S)
                    # for each distribution, resulting in the shape (L, )
                    entropy = torch.stack([d.entropy().mean() for d in dist])

                    # PPO ratio: exp(log pi_theta(a|s) - log pi_old(a|s))
                    # (S, L) -> (L, )
                    new_log_prob = torch.stack([dist[j].log_prob(sb.actions[:, j])
                                                for j in range(self.num_levels)],
                                               dim=1)
                    ratio = torch.exp(new_log_prob - sb.log_prob)

                    # Surrogate objectives (S, L)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage

                    # Mean over batch dimension, so (S, L) -> (L,)
                    policy_loss = -torch.min(surr1, surr2).mean(axis=0)

                    # Value loss with clipping
                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean(axis=0)

                    # Loss per level (L,)
                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    # Kullback-Leibler (KL) divergence
                    kl = (sb.log_prob - new_log_prob).mean(axis=0)

                    # Update batch values

                    batch_entropy += entropy.cpu().numpy()
                    batch_value += value.mean(axis=0).cpu().numpy()
                    batch_policy_loss += policy_loss.cpu().numpy()
                    batch_value_loss += value_loss.cpu().numpy()
                    batch_kl += kl.cpu().numpy()
                    batch_loss_tensor += loss

                    # Save detached memory for the future recurrence
                    # step. What is important is that memory is saved
                    # in the experience and never read back. It is
                    # taken from the experience only once, when
                    # recrrence window starts (see a few blocks above
                    # where memory is inited), which means that these
                    # updates never cross the recurrence or epoch
                    # boundary and do not influence the training of
                    # subsequent epochs.
                    #
                    # This is executed for ALL steps except the last
                    # step in the recurrence window
                    if self.acmodel.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_kl /= self.recurrence
                batch_loss_tensor /= self.recurrence

                # Update actor-critic

                if apply_update:
                    self.optimizer.zero_grad()
                    batch_loss_tensor.sum().backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(),
                                                               self.max_grad_norm).item()
                    self.optimizer.step()
                else:
                    grad_norm = 0.0

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_kls.append(batch_kl)
                log_grad_norms.append(grad_norm)

        # Log some values

        logs = {
            "entropy": numpy.mean(log_entropies, axis=0), # (L, )
            "value": numpy.mean(log_values, axis=0), # (L, )
            "policy_loss": numpy.mean(log_policy_losses, axis=0), # (L, )
            "value_loss": numpy.mean(log_value_losses, axis=0), # (L, )
            "kl": numpy.mean(log_kls, axis=0), # (L, )
            "grad_norm": numpy.mean(log_grad_norms),
        }

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_frames_per_proc != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
