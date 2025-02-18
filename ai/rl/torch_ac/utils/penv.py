import multiprocessing
import gymnasium as gym


def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            result = env.step(data)
            terminated, truncated = result[2:4]
            if terminated or truncated:
                # Be careful here - the last observation is returned
                # right after reset, not the actual observation that
                # causes termination. This should not cause any
                # harm for training because the algorithm does not
                # actually use the next observation when done=True.
                # See @ParallelEnv.step()
                obs, _ = env.reset()
                result = (obs,) + result[1:]

            conn.send(result)
        elif cmd == "reset":
            obs, info = env.reset(seed=data)
            conn.send((obs, info))
        else:
            raise NotImplementedError


class ParallelEnvPool:
    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."
        self.ctx = multiprocessing.get_context('fork')
        self.envs = envs
        self.locals = []
        for env in self.envs[1:]:
            local, remote = self.ctx.Pipe()
            self.locals.append(local)
            p = self.ctx.Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()


class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, penv_pool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = penv_pool

        # For bot environment spaces are missing
        if hasattr(self.p.envs[0], 'observation_space'):
            self.observation_space = self.p.envs[0].observation_space
        if hasattr(self.p.envs[0], 'action_space'):
            self.action_space = self.p.envs[0].action_space

    def reset(self, seeds=None):
        if seeds is not None:
            assert len(seeds) and len(seeds) <= len(self.p.envs)
        else:
            seeds = [None] * len(self.p.envs)

        for local, seed in zip(self.p.locals, seeds[1:]):
            local.send(("reset", seed))

        results = [self.p.envs[0].reset(seed=seeds[0])] + \
            [local.recv() for local, _ in zip(self.p.locals, seeds[1:])]
        return zip(*results)

    def step(self, actions):
        assert len(actions) and len(actions) <= len(self.p.envs)

        for local, action in zip(self.p.locals, actions[1:]):
            local.send(("step", action))

        result = self.p.envs[0].step(actions[0])
        terminated, truncated = result[2:4]
        if terminated or truncated:
            # See the comment in @worker above
            obs, _ = self.p.envs[0].reset()
            result = (obs,) + result[1:]

        return zip(*[result] + [local.recv() for local, _ in zip(self.p.locals, actions[1:])])

    def render(self):
        raise NotImplementedError
