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
    env = ManyEnvs(penv_pool)

    if acmodel.recurrent:
        memories = torch.zeros(num_envs, acmodel.memory_size, device=device)

    for offset in range(0, episodes, num_envs):
        num_envs = min(episodes - offset, num_envs)
        seeds = range(seed + offset, seed + offset + num_envs)
        many_obs, _ = env.reset(seeds=seeds)

        # (P, L) shape
        returns = np.zeros((num_envs, num_levels), dtype=float)
        num_frames = np.zeros((num_envs,), dtype=int)
        durations = np.zeros((num_envs,), dtype=float)
        not_yet_done = np.ones((num_envs,), dtype=bool)

        if return_obss_actions:
            all_obss = [[] for _ in range(num_envs)]
            all_actions = [[] for _ in range(num_envs)]

        ts = time.time()

        while np.any(not_yet_done):
            active_indices = np.flatnonzero(not_yet_done)
            with torch.no_grad():
                preprocessed_obss = preprocess_obss(many_obs, device=device)
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

            assert len(active_indices) == len(actions) == len(many_obs)
            assert actions.shape[1] == num_levels

            if return_obss_actions:
                for i, o, a in zip(active_indices, many_obs, actions):
                    all_obss[i].append(o)
                    all_actions[i].append(a)

            many_obs, _, terminated, truncated, info = env.step(actions, active_indices)
            reward = np.array([i['hierarchy/rewards'] for i in info ], dtype=np.float32)
            done = np.asarray(terminated) | np.asarray(truncated)

            if pause:
                time.sleep(pause)

            # For the next round keep only active observations
            many_obs = np.array(many_obs)[~done]
            just_done_indices = active_indices[done]

            if acmodel.recurrent:
                # Zero out what's ended
                memories[just_done_indices] = 0

            returns[active_indices] += reward
            num_frames[active_indices] += 1
            durations[just_done_indices] = time.time() - ts
            not_yet_done[just_done_indices] = False

        logs["num_frames_per_episode"].extend(num_frames.tolist())
        logs["return_per_episode"].extend(returns.tolist())
        logs["duration_per_episode"].extend(durations.tolist())
        logs["seed_per_episode"].extend(list(seeds))
        if return_obss_actions:
            logs["observations_per_episode"].extend(all_obss)
            logs["actions_per_episode"].extend(all_actions)

    return logs
