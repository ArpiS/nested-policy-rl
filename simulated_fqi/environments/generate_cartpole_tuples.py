import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from os.path import join as pjoin
from cartpole_regulator import CartPoleRegulatorEnv



# fg and bg have different mass in this case

def tuples(n_trajectories=100):

	bg_tuples = []
	fg_tuples = []

	for _ in range(n_trajectories):
		### Background
		cartpole = CartPoleRegulatorEnv()
		curr_bg_tuples = cartpole.generate_tuples(n_iter=101, group="background")
		bg_tuples.extend(curr_bg_tuples)

		## Foreground
		cartpole = CartPoleRegulatorEnv()
		curr_fg_tuples = cartpole.generate_tuples(n_iter=101, group="foreground")
		fg_tuples.extend(curr_fg_tuples)
		
	return bg_tuples, fg_tuples


if __name__ == "__main__":
	tuples()
	





