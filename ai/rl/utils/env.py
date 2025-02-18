import gymnasium as gym

def make_env(env_key, env_config):
    env = gym.make(env_key, env_config=env_config)
    env.reset(seed=env_config['seed'])
    return env
