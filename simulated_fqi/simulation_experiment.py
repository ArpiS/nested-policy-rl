import numpy as np
import os, sys
sys.path.append('environments/')
# from generate_pendulum_tuples import tuples as tuples_pendulum
# from generate_cartpole_tuples import tuples as tuples_cartpole
import numpy as np
import pandas as pd
import random
import pickle, os, csv, math, time, joblib
from joblib import Parallel, delayed
import datetime as dt
from datetime import date, datetime, timedelta
from collections import Counter
import copy as cp
import tqdm
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import log_loss, f1_score, precision_score, recall_score, accuracy_score
#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
import collections 
#import shap
import seaborn as sns
import random
from sklearn.linear_model import LinearRegression
np.seterr(all="ignore")
import matplotlib.pyplot as plt
import tqdm
import math
import statsmodels.api as sm
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import json
import util as util_fqi
import sys
sys.path.append('models/')
# from lmmfqi import LMMFQIagent
# from fqi import FQIagent
from nfqi import NFQIagent
# from cfqi import CFQIagent
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from os.path import join as pjoin
from pendulum import PendulumEnv
# from cartpole import CartPoleEnv
from cartpole import CartPoleRegulatorEnv


enviroment = "cartpole"

if enviroment == "pendulum":
	env = PendulumEnv()
	state_dim = 3
	unique_actions = np.array([-2, -1, 0, 1, 2])
elif enviroment == "cartpole":
	env = CartPoleRegulatorEnv()
	state_dim = 4
	unique_actions = np.array([0, 1])


# bg_tuples, fg_tuples = tuples(n_trajectories=400)
# all_tuples = bg_tuples + fg_tuples
# random.shuffle(all_tuples)
# split = 0.8
# train_tuples = all_tuples[:int(split*len(all_tuples))]
# test_tuples = all_tuples[int(split*len(all_tuples)):]

# training_set, test_set = util_fqi.construct_dicts(train_tuples, test_tuples)

agent = NFQIagent(unique_actions=unique_actions, gamma=0.5, state_dim=state_dim, iters=400)
Q_dist = agent.runNFQI()
# Q_dist = agent.runFQI(repeats=1)


# Run fitted model online on test data
cartpole = CartPoleRegulatorEnv()
curr_state = env.reset()

n_test_trajectories = 5

### FQI agent
trajectory_lengths_fqi = []
trajectory_length = 0
for _ in range(n_test_trajectories):
	done = False
	ii = 0
	env.reset()
	while not done and ii < 100:
		trajectory_length += 1

		# curr_action = agent.piE.predict(np.expand_dims(curr_state, 0))[0]
		curr_action = agent.predictNFQI(curr_state)
		print(curr_action)
		# import ipdb; ipdb.set_trace()

		if enviroment == "pendulum":
			curr_action = np.array([curr_action - 1])
		curr_state, curr_reward, done, _ = env.step(curr_action)

		if done:
			env.reset()
			trajectory_lengths_fqi.append(trajectory_length)
			trajectory_length = 0

		env.render()
		ii += 1

import ipdb; ipdb.set_trace()


### Random agent
cartpole = CartPoleRegulatorEnv()
curr_state = cartpole.reset()

trajectory_lengths_random = []

for _ in range(n_test_trajectories):
	done = False
	trajectory_length = 0
	while not done:
		trajectory_length += 1

		curr_action = cartpole.action_space.sample()
		curr_state, curr_reward, done, _ = cartpole.step(np.array([curr_action]))

		if done:
			cartpole.reset()
			trajectory_lengths_random.append(trajectory_length)
			trajectory_length = 0



results_df = pd.DataFrame({"fqi": trajectory_lengths_fqi, "random": trajectory_lengths_random})
results_df = pd.melt(results_df)
sns.boxplot(data=results_df, x="variable", y="value")
plt.xlabel("Agent")
plt.ylabel("Length of cartpole trajectories")
plt.show()
import ipdb; ipdb.set_trace()

### Oracle agent
for ii in range(1, n_test_iter):

	agent
	# Randomly sample an action
	a = pend.action_space.sample()

	# Perform the action
	s, cost, _, _ = pend.step(a)
	states[ii, :] = s
	costs[ii] = cost

import ipdb; ipdb.set_trace()



