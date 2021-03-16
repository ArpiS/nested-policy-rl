import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from os.path import join as pjoin
from pendulum import PendulumEnv



# fg and bg have different mass in this case

def tuples(n_trajectories=100):

	bg_tuples = []
	fg_tuples = []

	for _ in range(n_trajectories):
		### Background
		pend = PendulumEnv(m=1.0)
		curr_bg_tuples = pend.generate_tuples(n_iter=101, group="background")
		bg_tuples.extend(curr_bg_tuples)

		## Foreground
		pend = PendulumEnv(m=5.0)
		curr_fg_tuples = pend.generate_tuples(n_iter=101, group="foreground")
		fg_tuples.extend(curr_fg_tuples)
		
	return bg_tuples, fg_tuples


if __name__ == "__main__":
	tuples()
	





