"""
Transformer-based credit assigner for retrospective return decomposition.

CreditAssigner takes a window of LSTM hidden states (detached) and predicts
per-step contributions that should sum to the episode return.  Trained with:

    loss = MSE(sum(contributions), episode_return)
         + entropy_coef * H(contributions)
         + pos_coef    * positional_penalty(contributions)

The MSE term forces correct return reconstruction.
H is the entropy of the normalized absolute contributions - minimizing it
concentrates blame on few steps ("rich get richer" gradient, stable sparse
minimum, unlike L1 which has a flat loss surface for same-sign contributions).
The positional penalty is directional: for failure episodes (negative return)
it penalizes late contributions (pushing blame early); for success episodes
(positive return) it penalizes early contributions (pushing credit late).

Training uses cross-episode attention: N episodes (half success, half failure)
are concatenated into one sequence so the transformer can attend across episode
boundaries and learn which h-state patterns differentiate the two outcomes.
Inference uses single-episode mode (no cross-episode context needed).

CreditAssignerBuffer maintains separate queues for success and failure episodes
and returns balanced batches so every training group has both types.
"""

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class CreditAssigner(nn.Module):
    def __init__(self, hidden_dim, embed_dim=256, nhead=4, num_layers=2,
                 window=512, max_group=8):
        super().__init__()
        self.window    = window
        self.max_group = max_group
        self.proj      = nn.Linear(hidden_dim, embed_dim)
        # Within-episode positional embedding (shared across all episodes in a group)
        self.pos_embed = nn.Parameter(torch.zeros(window, embed_dim))
        # Episode-identity embedding for cross-episode training (max_group slots)
        self.ep_embed  = nn.Parameter(torch.zeros(max_group, embed_dim))
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead,
            dim_feedforward=embed_dim * 2,
            batch_first=True, dropout=0.0,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head    = nn.Linear(embed_dim, 1)

    def forward(self, states, episode_ids=None):
        """
        Training (cross-episode):
            states      : (1, N*W, hidden_dim)  - N episodes concatenated
            episode_ids : (N*W,)  int           - episode index per token (0..N-1)
            returns     : (1, N*W)

        Inference (single-episode):
            states      : (P, T, hidden_dim)    - P independent rollout chunks
            episode_ids : None
            returns     : (P, T)
        """
        T = states.shape[1]
        x = self.proj(states)                                      # (..., T, embed_dim)

        if episode_ids is not None:
            # Cross-episode: within-episode position repeats every W steps
            within = torch.arange(T, device=states.device) % self.window  # (T,)
            x = x + self.pos_embed[within].unsqueeze(0)            # (1, T, embed_dim)
            x = x + self.ep_embed[episode_ids].unsqueeze(0)        # (1, T, embed_dim)
        else:
            # Single-episode inference: use last T positions of pos_embed
            x = x + self.pos_embed[-T:].unsqueeze(0)               # (P, T, embed_dim)

        x = self.encoder(x)                                        # (..., T, embed_dim)
        return self.head(x).squeeze(-1)                            # (..., T)


class CreditAssignerBuffer:
    """
    Per-environment episode accumulator.

    Completed episodes are routed into two separate queues by outcome.
    pop_batch() returns a balanced batch: half success, half failure.
    This ensures every cross-episode training group sees both outcome types
    so the transformer can learn which h-state patterns differentiate them.
    """

    def __init__(self, num_envs, window, max_completed=512):
        self.window     = window
        self._states    = [deque(maxlen=window) for _ in range(num_envs)]
        self._returns   = [0.0] * num_envs
        # Separate queues; each holds up to max_completed episodes independently.
        # Do NOT share the capacity (// 2) - the two types arrive at different
        # rates and the faster-filling queue must be able to buffer while waiting
        # for the slower one.
        self._successes = deque(maxlen=max_completed)
        self._failures  = deque(maxlen=max_completed)

    def step(self, memories_h, rewards, dones, successes):
        """
        memories_h : (P, D) CPU tensor - h component of LSTM state at this step
        rewards    : (P,)  CPU tensor - shaped reward at this step
        dones      : (P,)  bool-like  - True if episode ended after this step
        successes  : (P,)  bool-like  - True if the finished episode was a success
                                        (only meaningful when dones[p] is True)
        """
        for p in range(len(dones)):
            self._states[p].append(memories_h[p].clone())
            self._returns[p] += float(rewards[p])
            if bool(dones[p]):
                buf = list(self._states[p])
                n   = len(buf)
                W   = self.window
                if n >= W:
                    window_states = torch.stack(buf)
                else:
                    pad = [torch.zeros_like(buf[0])] * (W - n)
                    window_states = torch.stack(pad + buf)          # zero-pad at START
                entry = (window_states, self._returns[p], n)
                if bool(successes[p]):
                    self._successes.append(entry)
                else:
                    self._failures.append(entry)
                self._states[p].clear()
                self._returns[p] = 0.0

    def pop_batch(self, batch_size):
        """
        Returns (states, returns, lengths, is_success) with batch_size//2
        successes followed by batch_size//2 failures, or None if either queue
        is too small.  is_success is a bool tensor of shape (B,); True for the
        first half (success), False for the second half (failure).
        """
        half = batch_size // 2
        if len(self._successes) < half or len(self._failures) < half:
            return None
        batch = ([self._successes.popleft() for _ in range(half)] +
                 [self._failures.popleft()  for _ in range(half)])
        states     = torch.stack([b[0] for b in batch])            # (B, W, D)
        returns    = torch.tensor([b[1] for b in batch], dtype=torch.float32)  # (B,)
        lengths    = torch.tensor([b[2] for b in batch], dtype=torch.long)     # (B,)
        is_success = torch.zeros(batch_size, dtype=torch.bool)
        is_success[:half] = True
        return states, returns, lengths, is_success

    @property
    def completed_count(self):
        return len(self._successes) + len(self._failures)

    @property
    def success_count(self):
        return len(self._successes)

    @property
    def failure_count(self):
        return len(self._failures)


def credit_assigner_loss(contributions, episode_returns, lengths,
                         sparsity_coef, pos_coef, is_success=None):
    """
    contributions    : (B, W) per-step scalars
    episode_returns  : (B,)   true episode returns
    lengths          : (B,)   actual episode lengths (<=W); padded positions masked
    sparsity_coef    : float  entropy sparsity weight (H term)
    pos_coef         : float  directional positional penalty weight
    is_success       : (B,)   bool, True = success episode (from buffer routing).
                               If None, falls back to episode_returns >= 0.

    Padding is at the START of each window (positions 0..W-n-1 are zeros).
    The mask selects positions >= W - length.

    Positional penalty direction:
      failure episodes: penalize late contributions  -> push blame early
      success episodes: penalize early contributions -> push credit late
    This creates temporal contrast: failures blamed early, successes credited late,
    which makes the cross-episode attention signal easier to exploit.
    """
    B, W = contributions.shape
    device = contributions.device

    # Boolean mask: True for real steps, False for zero-padded prefix
    first_real = W - lengths.to(device)                            # (B,)
    mask = torch.arange(W, device=device).unsqueeze(0) >= \
           first_real.unsqueeze(1)                                 # (B, W)
    mask_f = mask.float()

    pred_return = (contributions * mask_f).sum(dim=1)              # (B,)
    mse = F.mse_loss(pred_return, episode_returns)

    # Entropy sparsity over real steps only
    abs_c   = contributions.abs() * mask_f
    weights = abs_c / abs_c.sum(dim=1, keepdim=True).clamp(min=1e-8)
    entropy = -(weights * (weights + 1e-8).log()).sum(dim=1).mean()

    # Positional weight: 0.0 at first real step, 1.0 at last real step
    pos_idx = torch.arange(W, device=device).unsqueeze(0)         # (1, W)
    pos_w   = (pos_idx - first_real.unsqueeze(1)).float() / \
              (lengths.float().to(device) - 1).unsqueeze(1).clamp(min=1)
    pos_w   = pos_w.clamp(0.0, 1.0)

    # Use explicit success flags from buffer routing (reliable) if available;
    # fall back to return sign (can misclassify death-with-kills episodes).
    if is_success is None:
        suc = (episode_returns >= 0)
    else:
        suc = is_success.to(device)
    pos_w_dir   = torch.where(suc.unsqueeze(1), 1.0 - pos_w, pos_w)  # (B, W)
    pos_penalty = (abs_c * pos_w_dir * mask_f).sum(dim=1).mean()

    loss = mse + sparsity_coef * entropy + pos_coef * pos_penalty

    # Diagnostic: where does blame/credit concentrate?
    # 0.0 = first real step, 1.0 = last real step
    peak_global = abs_c.argmax(dim=1).float()                      # (B,)
    peak_rel    = (peak_global - first_real.float()) / \
                  (lengths.float().to(device) - 1).clamp(min=1)
    blame_pos   = peak_rel.mean()

    sum_abs_err = (pred_return - episode_returns).abs().mean()
    return loss, {"ca_loss":      loss.item(),
                  "ca_mse":       mse.item(),
                  "ca_entropy":   entropy.item(),
                  "ca_pos":       pos_penalty.item(),
                  "ca_sum_err":   sum_abs_err.item(),
                  "ca_blame_pos": blame_pos.item()}
