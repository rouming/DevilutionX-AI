import multiprocessing
import selectors


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
            obs, info = env.reset(seed=data)
            conn.send((obs, info))
        else:
            raise NotImplementedError


class ParallelEnvPool:
    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."
        self.sel = selectors.DefaultSelector()
        self.ctx = multiprocessing.get_context('fork')
        self.envs = envs
        self.conns = []
        for i, env in enumerate(self.envs):
            local, remote = self.ctx.Pipe()
            self.sel.register(local, selectors.EVENT_READ, i)
            self.conns.append(local)
            p = self.ctx.Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()


class ParallelEnv:
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, penv_pool, auto_reset=False):
        self.p = penv_pool
        self.auto_reset = auto_reset


    def nonblock_reset(self, seeds=None, indices=None):
        if seeds is None and indices is None:
            seeds = [None] * len(self.p.envs)
            indices = range(0, len(self.p.envs))
        elif seeds is None and indices is not None:
            assert len(indices)
            seeds = [None] * len(indices)
        elif seeds is not None and indices is None:
            assert len(seeds) == len(self.p.envs)
            indices = range(0, len(self.p.envs))
        elif seeds is not None and indices is not None:
            assert len(indices)
            assert len(seeds) == len(indices)
            assert len(indices) <= len(self.p.envs)
        else:
            assert 0, "Unknown seeds/indices combination"

        for i, ind in enumerate(indices):
            local, seed = self.p.conns[ind], seeds[i]
            local.send(("reset", seed))


    def poll_resets(self, *, nonblock):
        indices = []
        results = []
        timeout = 0 if nonblock else None
        events = self.p.sel.select(timeout=timeout)
        for e, mask in events:
            i = e.data
            conn = e.fileobj
            result = conn.recv()
            indices.append(i)
            results.append(result)

        obss, infos = zip(*results) if results else ((), ())
        return indices, obss, infos


    def reset(self, seeds=None, indices=None):
        obss = []
        infos = []
        self.nonblock_reset(seeds=seeds, indices=indices)
        while len(obss) != len(indices):
            _, obs, info = self.poll_resets(nonblock=False)
            obss.extend(obs)
            infos.extend(info)

        return obss, infos


    def step(self, actions, indices=None):
        if indices is None:
            assert len(actions) == len(self.p.envs)
            indices = range(0, len(self.p.envs))
        else:
            assert indices is not None and len(indices)
            assert len(actions) == len(indices)
            assert len(indices) <= len(self.p.envs)

        for i, ind in enumerate(indices):
            local, action = self.p.conns[ind], actions[i]
            local.send(("step", (action, self.auto_reset)))

        results = []
        for i, ind in enumerate(indices):
            local = self.p.conns[ind]
            result = local.recv()
            obs, reward, terminated, truncated, info = result

            result = obs, reward, terminated, truncated, info
            results.append(result)

        return zip(*results)


    def render(self):
        raise NotImplementedError
