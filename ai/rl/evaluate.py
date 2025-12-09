import numpy as np
import time
import torch

from rl.torch_ac.utils import ParallelEnv
from rl.utils import device

class ManyEnvs(ParallelEnv):
    def __init__(self, penv_pool, *args, **kwargs):
        super().__init__(penv_pool, *args, **kwargs)

    def reset(self, seeds=None):
        results = super().reset(seeds=seeds)
        return results

    def step(self, actions, active_indices):
        assert len(actions) == len(active_indices)

        for i, ind in enumerate(active_indices):
            if ind > 0:
                local, action = self.p.locals[ind - 1], actions[i]
                local.send(("step", action))

        results = []
        for i, ind in enumerate(active_indices):
            if ind == 0:
                result = self.p.envs[0].step(actions[i])
            else:
                local = self.p.locals[ind - 1]
                result = local.recv()
            results.append(result)

        return zip(*results)

    def render(self):
        raise NotImplementedError


# Evaluate the model with a specific number of episodes starting from
# a seed value
def batch_evaluate(acmodel, preprocess_obss, penv_pool, argmax, seed,
                   episodes, pause=0.0):
    logs = {
        "num_frames_per_episode": [],
        "return_per_episode": [],
        "duration_per_episode": [],
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

    seeds = np.arange(seed, seed + num_envs, dtype=int)
    max_seed = seed + episodes
    next_seed = seed + num_envs

    timestamps[:] = time.time()

    if acmodel.recurrent:
        memories = torch.zeros(num_envs, acmodel.memory_size, device=device)

    active_indices = np.flatnonzero(running_envs)
    obss, _ = env.ext_reset(seeds=seeds.tolist(), active_indices=active_indices)
    obss = np.asarray(obss)

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

        # Actions shape (P, L)
        if argmax:
            actions = torch.stack([d.probs.argmax(dim=1) for d in dist], dim=1)
        else:
            actions = torch.stack([d.sample() for d in dist], dim=1)

        actions = actions.cpu().numpy()

        assert len(active_indices) == len(actions) == len(obs)
        assert actions.shape[1] == num_levels

        obs, reward, terminated, truncated, _, _ = env.ext_step(actions, active_indices)
        done = np.asarray(terminated) | np.asarray(truncated)

        returns[active_indices] += reward
        obss[active_indices] = obs
        num_frames[active_indices] += 1

        just_done_indices = active_indices[done]

        if just_done_indices.size:
            done_num_frames = num_frames[just_done_indices]
            done_returns = returns[just_done_indices]
            done_durations = time.time() - timestamps[just_done_indices]
            done_seeds = seeds[just_done_indices]

            logs["num_frames_per_episode"].extend(done_num_frames.tolist())
            logs["return_per_episode"].extend(done_returns.tolist())
            logs["duration_per_episode"].extend(done_durations.tolist())
            logs["seed_per_episode"].extend(done_seeds.tolist())

            end_seed = min(max_seed, next_seed + just_done_indices.size)
            new_seeds = np.arange(next_seed, end_seed, dtype=int)
            next_seed = end_seed

            nr_restart = new_seeds.size

            restart_indices = just_done_indices[:nr_restart]
            finished_indices = just_done_indices[nr_restart:]
            running_envs[finished_indices] = False

            if restart_indices.size:
                reset_obs, _ = env.ext_reset(seeds=new_seeds.tolist(), active_indices=restart_indices)
                obss[restart_indices] = reset_obs
                seeds[restart_indices] = new_seeds
                timestamps[restart_indices] = time.time()
                num_frames[restart_indices] = 0
                returns[restart_indices] = 0

                if acmodel.recurrent:
                    memories[restart_indices] = 0

        if pause:
            time.sleep(pause)

    return logs
