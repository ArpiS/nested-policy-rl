# from gym_sepsis.envs.sepsis_env import SepsisEnv
from sepsis_env import SepsisEnv

env = SepsisEnv()
tuples = env.generate_tuples(group="foreground", n_trajectories=3)
