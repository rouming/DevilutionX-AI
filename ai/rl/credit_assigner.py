"""
Transformer-based credit assigner for retrospective return decomposition.

CreditAssigner takes a window of LSTM hidden states (detached) and predicts
per-step contributions that should sum to the episode return.  Trained with:

    loss = MSE(sum(contributions), episode_return) + lambda * H(contributions)

The MSE term forces correct return reconstruction; H is the entropy of the
normalized absolute contributions.  Minimizing entropy concentrates blame on
the actual bad decisions: the "rich get richer" gradient kills small
contributors aggressively while defending large ones, reaching a stable sparse
minimum unlike L1 which has a flat loss surface for same-sign contributions.

CreditAssignerBuffer accumulates per-environment (hidden_state, reward) pairs
and emits completed (states_window, episode_return) batches for training.
"""

import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class CreditAssigner(nn.Module):
    def __init__(self, hidden_dim, embed_dim=256, nhead=4, num_layers=2, window=512):
        super().__init__()
        self.window = window
        self.proj = nn.Linear(hidden_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(window, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead,
            dim_feedforward=embed_dim * 2,
            batch_first=True, dropout=0.0,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, states):
        # states: (B, T, hidden_dim) - LSTM h states, must be detached by caller
        # T may be < W (inference uses rollout chunks); take last T positions of
        # the embedding so inference steps align with the tail of the trained window.
        T = states.shape[1]
        x = self.proj(states) + self.pos_embed[-T:].unsqueeze(0)  # (B, T, embed_dim)
        x = self.encoder(x)                                        # (B, T, embed_dim)
        return self.head(x).squeeze(-1)                            # (B, T)


class CreditAssignerBuffer:
    """Per-environment episode accumulator."""

    def __init__(self, num_envs, window, max_completed=512):
        self.window = window
        # Ring buffer of hidden states per env (deque auto-drops oldest beyond window)
        self._states = [deque(maxlen=window) for _ in range(num_envs)]
        self._returns = [0.0] * num_envs
        self._completed = deque(maxlen=max_completed)

    def step(self, memories_h, rewards, dones):
        """
        memories_h : (P, D) CPU tensor - h component of LSTM state at this step
        rewards    : (P,)  CPU tensor - shaped reward at this step
        dones      : (P,)  bool - True if episode ended after this step
        """
        for p in range(len(dones)):
            self._states[p].append(memories_h[p].clone())
            self._returns[p] += float(rewards[p])
            if bool(dones[p]):
                buf = list(self._states[p])
                n = len(buf)
                W = self.window
                if n >= W:
                    window_states = torch.stack(buf)  # (W, D)
                else:
                    pad = [torch.zeros_like(buf[0])] * (W - n)
                    window_states = torch.stack(pad + buf)  # (W, D)
                self._completed.append((window_states, self._returns[p], n))
                self._states[p].clear()
                self._returns[p] = 0.0

    def pop_batch(self, batch_size):
        """Returns (states, returns, lengths) batch or None if not enough episodes yet."""
        if len(self._completed) < batch_size:
            return None
        batch = [self._completed.popleft() for _ in range(batch_size)]
        states  = torch.stack([b[0] for b in batch])                          # (B, W, D)
        returns = torch.tensor([b[1] for b in batch], dtype=torch.float32)    # (B,)
        lengths = torch.tensor([b[2] for b in batch], dtype=torch.long)       # (B,)
        return states, returns, lengths

    @property
    def completed_count(self):
        return len(self._completed)


def credit_assigner_loss(contributions, episode_returns, lengths, sparsity_coef):
    """
    contributions    : (B, W) per-step scalars
    episode_returns  : (B,)   true episode returns
    lengths          : (B,)   actual episode lengths (<=W); padded positions are masked
    sparsity_coef    : float  entropy sparsity weight

    Padding is at the START of each window (positions 0..W-n-1 are zeros,
    W-n..W-1 are real steps), so the mask selects positions >= W - length.
    Only real steps contribute to the sum and to the entropy penalty.

    Sparsity via entropy minimization: normalize |contributions| over real steps
    into a probability distribution and minimize its entropy.  Unlike L1, this
    has a "rich get richer" gradient - small contributions are killed aggressively
    while large ones are defended - and reaches a stable minimum at concentration.
    """
    B, W = contributions.shape
    # True for real steps, False for zero-padded prefix
    mask = torch.arange(W, device=contributions.device).unsqueeze(0) >= \
           (W - lengths.unsqueeze(1).to(contributions.device))      # (B, W)
    mask_f = mask.float()

    pred_return = (contributions * mask_f).sum(dim=1)               # (B,)
    mse = F.mse_loss(pred_return, episode_returns)

    abs_c = contributions.abs() * mask_f                            # (B, W) zero out padding
    weights = abs_c / abs_c.sum(dim=1, keepdim=True).clamp(min=1e-8)
    entropy = -(weights * (weights + 1e-8).log()).sum(dim=1).mean()
    loss = mse + sparsity_coef * entropy

    sum_abs_err = (pred_return - episode_returns).abs().mean()

    # Where does blame concentrate?  For each episode find the real-step index
    # (0 = first real step, 1 = last real step) that carries the most |contribution|.
    # If the transformer is lazy this will sit near 1.0 (always the death frame).
    first_real = W - lengths.to(contributions.device)          # (B,) index of first real step
    peak_global = (abs_c).argmax(dim=1).float()                # (B,) absolute W index
    peak_rel    = (peak_global - first_real) / (lengths.float().to(contributions.device) - 1).clamp(min=1)
    blame_pos   = peak_rel.mean()                              # 0=start, 1=last frame

    return loss, {"ca_loss": loss.item(),
                  "ca_mse": mse.item(),
                  "ca_entropy": entropy.item(),
                  "ca_sum_abs_err": sum_abs_err.item(),
                  "ca_blame_pos": blame_pos.item()}
