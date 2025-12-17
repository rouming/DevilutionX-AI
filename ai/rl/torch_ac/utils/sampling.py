import torch


def calculate_deterministic_noise(global_seed, env_seeds, env_resets, env_steps,
                                  dims):
    """
    Args:
        global_seed (int): Global experiment seed
        env_seeds (Tensor): (Batch,) Unique seeds for each environment runner
        env_resets (Tensor): (Batch,) Resets counter for each environment runner
        env_steps (Tensor): (Batch,) Step counter for each environment runner
        dims (int): Desired output number of dimensions per runner
    Returns:
        noise (Tensor): Shape (B, dims) -> e.g., (B, L)
    """
    assert((env_seeds.dtype, env_seeds.device, env_seeds.shape) ==
           (env_resets.dtype, env_resets.device, env_resets.shape) ==
           (env_steps.dtype, env_steps.device, env_steps.shape))

    B, L = env_seeds.shape[0], dims

    seeds = env_seeds.unsqueeze(1).expand(B, L)
    resets = env_resets.unsqueeze(1).expand(B, L)
    steps = env_steps.unsqueeze(1).expand(B, L)
    dim_ids = torch.arange(L, device=env_seeds.device).unsqueeze(0).expand(B, L)

    # Mix inputs using large primes
    # We use XOR (^) to combine the streams
    mixed_input = (seeds       * 0x1B873593) ^ \
                  (resets      * 0x85EBCA6B) ^ \
                  (steps       * 0x73856093) ^ \
                  (dim_ids     * 0xED558CCD) ^ \
                  (global_seed * 0x9E3779B9)

    # MurmurHash3 Finalizer
    x = mixed_input
    x = (x ^ (x >> 16)) * 0x7feb352d
    x = (x ^ (x >> 15)) * 0x846ca68b
    x = x ^ (x >> 16)

    # Output float [0, 1), (B, L) shape
    return (x & 0xFFFFFFFF).float() / (2**32)


def deterministic_sample(probs, noise):
    """
    probs: (B, A)
    noise: (B,)
    """
    # Inverse CDF sampling
    # Unsqueeze noise to (B, 1) for broadcasting
    cdf = probs.cumsum(dim=-1)
    actions = torch.searchsorted(cdf, noise.unsqueeze(-1)).squeeze(-1)

    # Safety clamp for float edge cases
    return actions.clamp(max=probs.size(-1) - 1)
