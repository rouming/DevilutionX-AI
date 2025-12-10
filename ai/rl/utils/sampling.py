import torch


def deterministic_sample(probs, seed, env_seeds, env_resets, env_steps):
    """
    Samples actions deterministically using stateless hashing.

    Args:
        probs (Tensor): (Batch, Actions) probabilities summing to 1.
        seed (int): Global experiment seed.
        env_seeds (Tensor): (Batch,) Unique seeds for each environment runner.
        env_resets (Tensor): (Batch,) Resets counter for each environment runner.
        env_steps (Tensor): (Batch,) Step counter for each environment runner.

    Returns:
        actions (Tensor): (Batch,) Selected action indices.
    """
    # Mix inputs using large primes
    # We use XOR (^) to combine the streams
    mixed_input = (env_seeds  * 0x1B873593) ^ \
                  (env_resets * 0x85EBCA6B) ^ \
                  (env_steps  * 0x73856093) ^ \
                  (seed       * 0x9E3779B9)

    # MurmurHash3 Finalizer
    x = mixed_input
    x = (x ^ (x >> 16)) * 0x7feb352d
    x = (x ^ (x >> 15)) * 0x846ca68b
    x = x ^ (x >> 16)

    # Output float [0, 1)
    noise = (x & 0xFFFFFFFF).float() / (2**32)

    # Inverse CDF Sampling
    # Unsqueeze noise to (Batch, 1) for broadcasting
    cdf = probs.cumsum(dim=-1)
    actions = torch.searchsorted(cdf, noise.unsqueeze(-1)).squeeze(-1)

    # Safety clamp for float edge cases
    return actions.clamp(max=probs.size(-1) - 1)
