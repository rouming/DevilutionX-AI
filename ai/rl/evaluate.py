import numpy as np
import time
import torch

from rl.torch_ac.utils import ParallelEnv
from rl.utils import device
from rl.utils import deterministic_sample


# Evaluate the model with a specific number of episodes starting from
# a seed value
def batch_evaluate(acmodel, preprocess_obss, penv_pool, argmax, seed,
                   episodes, return_obss_actions=False, pause=0.0):
    logs = {
        "num_frames_per_episode": [],
        "return_per_episode": [],
        "duration_per_episode": [],
        "observations_per_episode": [],
        "actions_per_episode": [],
        "seed_per_episode": []
    }

    num_envs = min(len(penv_pool.envs), episodes)
    num_levels = penv_pool.envs[0].unwrapped.num_levels
    env = ParallelEnv(penv_pool)

    # (P, L) shape
    returns = np.zeros((num_envs, num_levels), dtype=float)
    # (P, ) shape
    num_frames = np.zeros((num_envs,), dtype=int)
    timestamps = np.zeros((num_envs,), dtype=float)
    running_envs = np.ones((num_envs,), dtype=bool)

    seeds = torch.arange(seed, seed + num_envs, dtype=int, device=device)
    max_seed = seed + episodes
    next_seed = seed + num_envs

    if return_obss_actions:
        log_obss = [[] for _ in range(num_envs)]
        log_actions = [[] for _ in range(num_envs)]

    timestamps[:] = time.time()

    if acmodel.recurrent:
        memories = torch.zeros(num_envs, acmodel.memory_size, device=device)

    active_indices = np.flatnonzero(running_envs)
    obss, info = env.ext_reset(seeds=seeds.tolist(), indices=active_indices)
    obss = np.asarray(obss)
    if not argmax:
        stats = torch.tensor([inf["stats"] for inf in info], dtype=int, device=device)

    while np.any(running_envs):
        active_indices = np.flatnonzero(running_envs)
        obs = obss[active_indices]
        with torch.no_grad():
            preprocessed_obss = preprocess_obss(obs, device=device)
            if acmodel.recurrent:
                memory = memories[active_indices]
                dist, _, memory = acmodel(preprocessed_obss, memory)
                memories[active_indices] = memory
            else:
                dist, _ = acmodel(preprocessed_obss)

            assert len(dist) == num_levels

        # Distributions shape (L, P) -> actions shape (P, L)
        if argmax:
            actions = torch.stack([d.probs.argmax(dim=1) for d in dist], dim=1)
        else:
            # We use stateless and deterministic categorical sampling
            aseeds = seeds[active_indices]
            astats = stats[active_indices]
            astats = (astats[:, 0], astats[:, 1])
            actions = torch.stack([deterministic_sample(d.probs, seed, aseeds, *astats)
                                   for d in dist], dim=1)

        actions = actions.cpu().numpy()

        assert len(active_indices) == len(actions) == len(obs)
        assert actions.shape[1] == num_levels

        if return_obss_actions:
            for i, o, a in zip(active_indices, obs, actions):
                log_obss[i].append(o)
                log_actions[i].append(a)

        obs, _, terminated, truncated, info = env.ext_step(actions, active_indices)
        done = np.logical_or(terminated, truncated)
        # HRL-aware rewards with the shape (P, L)
        reward = np.array([inf["hierarchy/reward"] for inf in info], dtype=float)
        assert reward.shape == (len(actions), num_levels)

        returns[active_indices] += reward
        obss[active_indices] = obs
        num_frames[active_indices] += 1
        if not argmax:
            st = torch.tensor([inf["stats"] for inf in info], dtype=int, device=device)
            stats[active_indices] = st

        if np.any(done):
            just_done_indices = active_indices[done]
            done_num_frames = num_frames[just_done_indices]
            done_returns = returns[just_done_indices]
            done_durations = time.time() - timestamps[just_done_indices]
            done_seeds = seeds[just_done_indices]

            logs["num_frames_per_episode"].extend(done_num_frames.tolist())
            logs["return_per_episode"].extend(done_returns.tolist())
            logs["duration_per_episode"].extend(done_durations.tolist())
            logs["seed_per_episode"].extend(done_seeds.tolist())

            if return_obss_actions:
                for i in just_done_indices:
                    logs["observations_per_episode"].append(log_obss[i])
                    logs["actions_per_episode"].append(log_actions[i])
                    log_obss[i] = []
                    log_actions[i] = []

            end_seed = min(max_seed, next_seed + len(just_done_indices))
            new_seeds = torch.arange(next_seed, end_seed, dtype=int, device=device)
            next_seed = end_seed

            nr_restart = len(new_seeds)

            restart_indices = just_done_indices[:nr_restart]
            finished_indices = just_done_indices[nr_restart:]
            running_envs[finished_indices] = False

            if len(restart_indices):
                new_obs, info = env.ext_reset(seeds=new_seeds.tolist(), indices=restart_indices)

                if not argmax:
                    new_stats = torch.tensor([inf["stats"] for inf in info], dtype=int, device=device)
                    stats[restart_indices] = new_stats

                obss[restart_indices] = new_obs
                seeds[restart_indices] = new_seeds
                timestamps[restart_indices] = time.time()
                num_frames[restart_indices] = 0
                returns[restart_indices] = 0

                if acmodel.recurrent:
                    memories[restart_indices] = 0

        if pause:
            time.sleep(pause)

    # Keep all logs sorted by seed
    order = np.argsort(logs["seed_per_episode"])
    for key, value in logs.items():
        if len(value):
            logs[key] = [value[i] for i in order]

    return logs
