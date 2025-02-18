import numpy as np
import time

from rl.torch_ac.utils import ParallelEnv

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


# Returns the performance of the agent on the environment for a
# particular number of episodes.
def batch_evaluate(agent, penv_pool, seed, episodes,
                   return_obss_actions=False, pause=0.0):
    logs = {
        "num_frames_per_episode": [],
        "return_per_episode": [],
        "duration_per_episode": [],
        "observations_per_episode": [],
        "actions_per_episode": [],
        "seed_per_episode": []
    }

    num_envs = min(len(penv_pool.envs), episodes)
    env = ManyEnvs(penv_pool)

    for offset in range(0, episodes, num_envs):
        num_envs = min(episodes - offset, num_envs)
        seeds = range(seed + offset, seed + offset + num_envs)
        many_obs, _ = env.reset(seeds=seeds)

        num_frames = np.zeros((num_envs,), dtype=int)
        durations = np.zeros((num_envs,), dtype=float)
        returns = np.zeros((num_envs,))
        not_yet_done = np.ones((num_envs,), dtype=bool)

        if return_obss_actions:
            obss = [[] for _ in range(num_envs)]
            actions = [[] for _ in range(num_envs)]

        ts = time.time()

        while np.any(not_yet_done):
            active_indices = np.flatnonzero(not_yet_done)
            actions = agent.get_actions(many_obs)
            assert len(active_indices) == len(actions) == len(many_obs)

            if return_obss_actions:
                for i, o, a in zip(active_indices, many_obs, actions):
                        obss[i].append(o)
                        actions[i].append(a)

            many_obs, reward, terminated, truncated, _ = env.step(actions, active_indices)
            done = np.asarray(terminated) | np.asarray(truncated)
            agent.analyze_feedbacks(reward, done)

            if pause:
                time.sleep(pause)

            # For the next round keep only active observations
            many_obs = np.array(many_obs)[~done]
            just_done_indices = active_indices[done]

            returns[active_indices] += reward
            num_frames[active_indices] += 1
            durations[just_done_indices] = time.time() - ts
            not_yet_done[just_done_indices] = False

        logs["num_frames_per_episode"].extend(list(num_frames))
        logs["return_per_episode"].extend(list(returns))
        logs["duration_per_episode"].extend(list(durations))
        logs["seed_per_episode"].extend(list(seeds))
        if return_obss_actions:
            logs["observations_per_episode"].extend(obss)
            logs["actions_per_episode"].extend(actions)

    return logs
