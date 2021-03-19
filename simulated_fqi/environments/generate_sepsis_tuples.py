from gym_sepsis.envs.sepsis_env import SepsisEnv
import numpy as np
import pandas as pd

N_STEPS = 10
N_ACTIONS = 25
ACTION_LIST = np.arange(N_ACTIONS)

env = SepsisEnv()

tuples = env.generate_tuples(group="foreground", n_trajectories=3)
import ipdb; ipdb.set_trace()


# history = pd.DataFrame(np.zeros((N_STEPS, len(env.features))), columns=env.features)

# for ii in range(N_STEPS):

# 	# Choose random action
# 	action = np.random.choice(ACTION_LIST)
	
# 	# Take action
# 	env.step(action)

# 	# Save current state
# 	df = env.render()
# 	history.iloc[ii, :] = df.iloc[0, :]

# df = env.render()
# import ipdb; ipdb.set_trace()