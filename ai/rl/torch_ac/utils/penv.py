import multiprocessing

from rl import utils

def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            action, auto_reset = data
            result = env.step(action)
            if auto_reset:
                obs, reward, terminated, truncated, info = result
                if terminated or truncated:
                    # Be careful here - the last observation is returned
                    # right after reset, not the actual observation that
                    # causes termination. This should not cause any
                    # harm for training because the algorithm does not
                    # actually use the next observation when done=True.
                    # See @ParallelEnv.step()
                    obs, reset_info = env.reset()
                    info |= reset_info
                    result = (obs, reward, terminated, truncated, info)

            conn.send(result)
        elif cmd == "reset":
            seed = data
            if seed is not None:
                utils.seed(seed)
            obs, info = env.reset(seed=seed)
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


class ParallelEnv:
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, penv_pool, auto_reset=False):
        self.p = penv_pool
        self.auto_reset = auto_reset


    def ext_reset(self, seeds=None, active_indices=None):
        if seeds is None and active_indices is None:
            seeds = [None] * len(self.p.envs)
            active_indices = range(0, len(self.p.envs))
        elif seeds is None and active_indices is not None:
            assert len(active_indices)
            seeds = [None] * len(active_indices)
        elif seeds is not None and active_indices is None:
            assert len(seeds) == len(self.p.envs)
            active_indices = range(0, len(self.p.envs))
        elif seeds is not None and active_indices is not None:
            assert len(active_indices)
            assert len(seeds) == len(active_indices)
            assert len(active_indices) <= len(self.p.envs)
        else:
            assert 0, "Unknown seeds/active_indices combination"

        for i, ind in enumerate(active_indices):
            if ind > 0:
                local, seed = self.p.locals[ind - 1], seeds[i]
                local.send(("reset", seed))

        results = []
        for i, ind in enumerate(active_indices):
            if ind == 0:
                seed = seeds[i]
                if seed is not None:
                    utils.seed(seed)
                result = self.p.envs[0].reset(seed=seed)
            else:
                local = self.p.locals[ind - 1]
                result = local.recv()
            results.append(result)

        return zip(*results)


    def ext_step(self, actions, active_indices=None):
        if active_indices is None:
            assert len(actions) == len(self.p.envs)
            active_indices = range(0, len(self.p.envs))
        else:
            assert active_indices is not None and len(active_indices)
            assert len(actions) == len(active_indices)
            assert len(active_indices) <= len(self.p.envs)

        for i, ind in enumerate(active_indices):
            if ind > 0:
                local, action = self.p.locals[ind - 1], actions[i]
                local.send(("step", (action, self.auto_reset)))

        results = []
        for i, ind in enumerate(active_indices):
            if ind == 0:
                result = self.p.envs[0].step(actions[i])
                obs, reward, terminated, truncated, info = result
                if self.auto_reset and (terminated or truncated):
                    # See the comment in @worker above
                    obs, reset_info = self.p.envs[0].reset()
                    info |= reset_info
            else:
                local = self.p.locals[ind - 1]
                result = local.recv()
                obs, reward, terminated, truncated, info = result

            result = obs, reward, terminated, truncated, info
            results.append(result)

        return zip(*results)


    def render(self):
        raise NotImplementedError
