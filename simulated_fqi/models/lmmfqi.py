import numpy as np
import pandas as pd
import pickle, os, csv, math, time
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
import sys
sys.path.append('../models/')
from lmm import LMM
import util as util_fqi


# def a2c(action):
# 	actions = [[0, 0], [0, 1], [1, 0], [1, 1]]
# 	#actions = [[0, 0], [0, 2], [3, 1], [4, 4]]
# 	classes = []
# 	for a in action:
# 		a = list(a)
# 		for c in range(len(actions)):
# 			if actions[c] == a:
# 				classes.append(c)
# 	return classes

# def p2c(pred):
# 	if pred <= 0.25:
# 		action = [0, 0]
# 	elif pred <= 0.5:
# 		action = [0, 1]
# 	elif pred <= 0.75:
# 		action = [1, 0]
# 	else:
# 		action = [1, 1]

# # Mapping states to actions?
# def c2a(c):
# 	d = {0: [0, 0], 1: [0, 1], 2: [1, 0], 3: [1, 1]}
# 	return np.array([d[k] for k in c])

# def random_weights(size=5):
# 	# w = 2*np.random.uniform(size=size) - 1
# 	w = norm(np.random.uniform(size=size))
# 	# w / np.sum(np.abs(w))

# 	return w

# def norm(vec):
# 	return vec / np.sum(np.abs(vec))

# def learnBehaviour(training_set, test_set):
# 	floc = "behavior.pkl"
# 	# if os.path.exists(floc):
# 	#    behaviour_pi = pickle.load(open(floc, 'rb'))
# 	# else:
# 	## Use a linear regression to predict behavior
# 	behaviour_pi = LinearRegression()
# 	X = np.vstack((training_set['s'], test_set['s']))
# 	X = np.reshape(X, (-1, 10))
# 	print(str(X.shape))
# 	y = a2c(np.vstack((training_set['a'], test_set['a'])))
# 	print(str(len(y)))
# 	behaviour_pi.fit(X, y)
# 	pickle.dump(behaviour_pi, open(floc, 'wb'))

# 	return behaviour_pi

# def construct_dicts(train_tuples, test_tuples):
# 	train = {}
# 	test = {}
# 	elts = ['s', 'a', 'ns', 'r', 'ds', 'vnum']
# 	for elt in elts:
# 		train[elt] = []
# 		test[elt] = []

# 	for tup in train_tuples:
# 		train['s'].append(tup[0])
# 		a = tup[1]
# 		try:
# 			a = np.concatenate(a).ravel()
# 			a = list(a)
# 			train['a'].append(a)
# 		except:
# 			train['a'].append(a)
# 		train['ns'].append(tup[2])
# 		train['r'].append(tup[3])
# 		train['ds'].append(tup[4])
# 		train['vnum'].append(tup[5])

# 	for tup in test_tuples:
# 		test['s'].append(tup[0])
# 		try:
# 			a = tup[1]
# 			a = np.concatenate(a).ravel()
# 			a = list(a)
# 			test['a'].append(a)
# 		except:
# 			test['a'].append(tup[1])
# 		test['ns'].append(tup[2])
# 		test['r'].append(tup[3])
# 		test['ds'].append(tup[4])
# 		test['vnum'].append(tup[5])
# 	return train, test

# shape, scale = 2, 10
# transition_foreground = np.random.gamma(shape, scale, (12, 10))

# mu, sigma = 0, 4 # mean and standard deviation
# transition_background = np.random.normal(mu, sigma, (12, 10))

# mu, sigma = 0, 5
# reward_function = np.random.normal(mu, sigma, (12, 1))

# exploit = 0.6
# explore = 1-exploit
# num_samples = 100
# num_patients = 100
# actions = [[0, 0], [0, 1], [1, 0], [1, 1]]
# mu, sigma = 0, 4

# transition_tuples = []
# for k, pat in enumerate(tqdm.tqdm(range(num_patients))):

# 	flip = np.random.choice(2)
# 	if flip == 0:
# 		ds = 'foreground'
# 	else:
# 		ds = 'background'
# 	# Generate a random initial state
# 	s = np.random.normal(mu, sigma, (10, 1))

# 	# Generate all of the tuples for this patient
# 	for i in range(num_samples):
# 		flip = random.uniform(0, 1)
# 		# Exploit
# 		if flip < exploit:
# 			all_rewards = []
# 			for j, a in enumerate(actions):
# 				a = np.asarray(a)
# 				a = np.reshape(a, (2, 1))
# 				s_a = np.concatenate((s, a))
# 				reward = np.dot(reward_function.T, s_a)
# 				all_rewards.append(reward)

# 			noise = np.random.normal(0, 0.05, 1)
# 			all_rewards = np.asarray(all_rewards)
# 			a = actions[np.argmax(all_rewards)]
# 			reward = np.max(all_rewards) + noise

# 			if ds == 'foreground':
# 				t_m = transition_foreground
# 			else:
# 				t_m = transition_background
# 			ns = np.matmul(s_a.T, t_m) / np.linalg.norm(np.matmul(s_a.T, t_m), ord=2)
# 			ns = np.add(ns, np.random.normal(0, 0.5, (1, 10)))  # Add noise


# 		# Explore
# 		else:
# 			a = np.asarray(actions[np.random.choice(3)])
# 			a = np.reshape(a, (2, 1))
# 			s_a = np.concatenate((s, a))  # concatenate the state and action

# 			if ds == 'foreground':
# 				t_m = transition_foreground
# 			else:
# 				t_m = transition_background
# 			ns = np.matmul(s_a.T, t_m) / np.linalg.norm(np.matmul(s_a.T, t_m), ord=2)
# 			ns = np.add(ns, np.random.normal(0, 0.5, (1, 10)))  # Add noise

# 			reward = np.dot(reward_function.T, s_a) + np.random.normal(0, 0.5, 1)

# 		# Transition tuple includes state, action, next state, reward, ds
# 		transition_tuples.append((list(s.flatten()), list(a), list(ns.flatten()), reward.flatten(), ds, i))
# 		s = ns.T

# split = int(0.8*len(transition_tuples))
# train_tuples = transition_tuples[:split]
# test_tuples = transition_tuples[split:]


class LMMFQIagent():
	def __init__(self, train_tuples, test_tuples, iters=150, gamma=0.99, batch_size=100, prioritize=False, estimator='lin',
				 weights=np.array([1, 1, 1, 1, 1]) / 5., maxT=36):

		self.iters = iters
		self.gamma = gamma
		self.batch_size = batch_size
		self.prioritize_a = prioritize
		self.training_set, self.test_set = util_fqi.construct_dicts(train_tuples, test_tuples)
		self.raw_test = test_tuples

		self.visits = {'train': len(train_tuples), 'test': len(test_tuples)}
		self.NV = {'train': len(train_tuples), 'test': len(test_tuples)}
		self.n_samples = len(self.training_set['s'])
		_, self.unique_actions, self.action_counts, _ = self.sub_actions()
		self.state_feats = [str(x) for x in range(10)]
		self.n_features = len(self.state_feats)
		self.reward_weights = weights
		self.maxT = maxT
		self.piB = util_fqi.learnBehaviour(self.training_set, self.test_set)
		self.n_actions = 4

		self.q_est = LMM()

		self.piE = LinearRegression()
		self.eval_est = LGBMRegressor(n_estimators=50, silent=True)

	def sub_actions(self):

		a = self.training_set['a']
		a = list(a)

		unique_actions = 0
		action_counts = 0
		n_actions = 0

		unique_actions, action_counts = np.unique(a, axis=0, return_counts=True)
		n_actions = len(unique_actions)

		return a, unique_actions, action_counts, n_actions

	def sampleTuples(self):
		ids = list(np.random.choice(np.arange(self.n_samples), self.batch_size, replace=False))
		batch = {}
		for k in self.training_set.keys():
			batch[k] = np.asarray(self.training_set[k], dtype=object)[ids]
		batch['r'] = np.dot(batch['r'] * [1, 1, 10, 10, 100], self.reward_weights)
		batch['s_ids'] = np.asarray(ids, dtype=int)
		batch['ns_ids'] = np.asarray(ids, dtype=int) + 1

		return batch

	def fitQ(self, batch, Q):

		batch_foreground = {}
		batch_background = {}
		groups = []
		elts = ['s', 'a', 'ns', 'r', 'ds', 'vnum', 's_ids', 'ns_ids']
		for el in elts:
			batch_foreground[el] = []
			batch_background[el] = []

		for i in range(len(batch['s_ids'])):
			if batch['ds'][i] == 'foreground':
				for k in batch.keys():
					batch_foreground[k].append(batch[k][i])
			else:
				for k in batch.keys():
					batch_background[k].append(batch[k][i])

		for i in range(len(batch['s_ids'])):
			if batch['ds'][i] == 'foreground':
				groups.append(1)
			else:
				groups.append(0)
		# input = [state action]
		x_shared = np.hstack((batch['s'], batch['a']))
		y_shared = batch['r'] + (self.gamma * np.max(Q[batch['ns_ids'], :], axis=1))

		groups = np.expand_dims(groups, axis=1)
		self.q_est.fit(x_shared, y_shared, groups)

		return batch_foreground, batch_background

	def updateQtable(self, Qtable, batch_fg, batch_bg):
		# Update for foregound using just foreground
		# Update for background using shared

		bg_size = len(batch_bg['s'])
		fg_size = len(batch_fg['s'])
		for i, a in enumerate(self.unique_actions):
			Qtable[batch_bg['s_ids'], i] = self.q_est.predict(np.hstack((batch_bg['ns'], np.tile(a, (bg_size, 1)))), np.tile([0], (bg_size, 1)))
			Qtable[batch_fg['s_ids'], i] = self.q_est.predict(np.hstack((batch_fg['ns'], np.tile(a, (fg_size, 1)))), np.tile([1], (fg_size, 1)))
		return Qtable

	def runFQI(self, repeats=10):

		print('Learning policy')
		meanQtable = np.zeros((self.n_samples + 1, self.n_actions))

		for r in range(repeats):
			print('Run', r, ':')
			print('Initialize: get batch, set initial Q')
			Qtable = np.zeros((self.n_samples + 1, self.n_actions))
			Qdist = []

			# print('Run FQI')
			for iteration in range(self.iters):
				# copy q-table
				Qold = cp.deepcopy(Qtable)

				# sample batch
				batch = self.sampleTuples()

				# learn q_est with samples, targets from batch
				batch_foreground, batch_background = self.fitQ(batch, Qtable)

				# update Q table for all s given new estimator
				self.updateQtable(Qtable, batch_foreground, batch_background)

				# check divergence from last estimate
				Qdist.append(mean_absolute_error(Qold, Qtable))

			# plt.plot(Qdist)
			meanQtable += Qtable

		meanQtable = meanQtable / repeats
		print('Learn policy')

		# Since the Q table is constructed contrastively, the policy is contrastive?
		self.getPi(meanQtable)
		return Qdist

	def getPi(self, Qtable):
		optA = np.argmax(Qtable, axis=1)
		print("Opta: ", optA)
		# print("Fitting to training set")
		# print("Optimal actions: ", optA)
		self.piE.fit(self.training_set['s'], optA[:-1])

# print("Done Fitting")

if __name__ == "__main__":
	lmmfqi_agent = LMMFQIagent(train_tuples=train_tuples, test_tuples=test_tuples)
	Q_dist = lmmfqi_agent.runFQI(repeats=1)
	# plt.plot(Q_dist, label= "Mixed Model FQI")
	# plt.xlabel("Iteration")
	# plt.ylabel("Q Estimate")
	# plt.legend()
	# plt.show()