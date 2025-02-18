import gymnasium as gym

def get_env_class(env_key):
    spec = gym.spec(env_key)
    return spec.entry_point

def make_env(env_key, env_config, game):
    return gym.make(env_key, env_config=env_config, game=game)
