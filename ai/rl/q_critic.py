"""
QR-DQN distributional Q-critic for HRL policy improvement.

QCritic takes the LSTM h-state (h only, not cell state) and outputs per-action
quantile values for each hierarchy level. N quantile values per action replace
the single scalar V(s) in standard PPO.

No fixed atom range required - quantile values adapt to the actual return scale
automatically. Changing reward magnitudes never breaks the critic silently.

Training: on-policy distributional TD with the QR-DQN quantile regression loss.
  target_t = r_t + gamma * Q(h_{t+1}, a_{t+1}) * not_done_t
  loss = qr_loss(Q(h_t, a_t), target_t, tau)

Actor update (Phase 2, --enable-q-critic):
  A(s, a) = mean(Q_quantiles(s, a)) - sum_a' pi(a'|s) * mean(Q_quantiles(s, a'))
  Plugs into PPO surrogate in place of GAE advantage.

Usage phases:
  Phase 1 (--warmup-q-critic):
      Main model frozen. Q-critic trains on rollouts from frozen policy.
      Q values converge before any policy change.
  Phase 2 (--enable-q-critic):
      Q-critic advantage replaces GAE in PPO actor update.
      Q-critic continues training each rollout.
"""

import torch
import torch.nn as nn


class QCritic(nn.Module):
    def __init__(self, hidden_dim, num_actions_per_level, num_quantiles=32):
        """
        hidden_dim            : LSTM h-state size (semi_memory_size)
        num_actions_per_level : list[int], one entry per hierarchy level
        num_quantiles         : N quantile values per action
        """
        super().__init__()
        self.num_quantiles = num_quantiles
        self.num_actions   = num_actions_per_level

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.ReLU(),
                nn.Linear(256, n_a * num_quantiles),
            )
            for n_a in num_actions_per_level
        ])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

        N = num_quantiles
        tau = torch.arange(1, 2 * N, 2, dtype=torch.float32) / (2 * N)
        self.register_buffer("tau", tau)    # (N,)

    def forward(self, h):
        """
        h       : (B, hidden_dim)
        returns : list of (B, num_actions_l, N) per level
        """
        out = []
        for l, head in enumerate(self.heads):
            x = head(h).view(h.shape[0], self.num_actions[l], self.num_quantiles)
            out.append(x)
        return out

    def forward_level(self, h, level):
        """Single-level forward for training loop efficiency."""
        x = self.heads[level](h)
        return x.view(h.shape[0], self.num_actions[level], self.num_quantiles)


def qr_loss(pred, target, tau):
    """
    QR-DQN quantile regression loss.

    pred   : (B, N) - predicted quantiles for the taken action
    target : (B, N) - target quantiles (must be detached)
    tau    : (N,)   - quantile midpoints

    Returns scalar loss (mean over all B, N_pred, N_target combinations).
    """
    # td[b, i, j] = target[b, j] - pred[b, i]
    td = target.unsqueeze(1) - pred.unsqueeze(2)            # (B, N, N)
    huber = torch.where(td.abs() <= 1.0,
                        0.5 * td.pow(2),
                        td.abs() - 0.5)
    weight = (tau.view(1, -1, 1) - (td < 0).float()).abs() # (B, N, N)
    return (huber * weight).mean()
