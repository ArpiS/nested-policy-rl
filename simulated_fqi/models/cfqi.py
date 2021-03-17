import numpy as np
import pandas as pd
import pickle, os, csv, math, time, joblib
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import util as util_fqi
import copy as cp

class CFQIagent():
    def __init__(self, train_tuples, test_tuples, iters=150, gamma=0.99, batch_size=100, prioritize=False, estimator='lin',
                 weights=np.array([1, 1, 1, 1, 1])/5., maxT=36, state_dim=10):
        
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
        self.piB = util_fqi.learnBehaviour(self.training_set, self.test_set, state_dim=state_dim)
        self.n_actions = len(self.unique_actions)
        
        if estimator == 'tree':
            self.q_est = ExtraTreesRegressor(n_estimators=50, max_depth=None, min_samples_leaf=10, min_samples_split=2,
                                             random_state=0)
        elif estimator == 'gbm':
            self.q_est = LGBMRegressor(n_estimators=50, silent=True)

        elif estimator == 'nn':
            self.q_est = None
        
        elif estimator == 'lin':
            self.q_est_shared = LinearRegression()
            self.q_est_fg = LinearRegression()
            
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
        
        # Divide into foreground and background batches. 
        batch_foreground = {}
        batch_background = {}
        
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
            
        # input = [state action]
        x_fg =  np.hstack((np.asarray(batch_foreground['s']), np.expand_dims(np.asarray(batch_foreground['a']), 1)))
        x_shared =  np.hstack((np.asarray(batch['s']), np.expand_dims(np.asarray(batch['a']), 1)))
        
        # target = r + gamma * max_a(Q(s', a))      == r for first iteration
        y_fg = batch_foreground['r'] + (self.gamma * np.max(Q[batch_foreground['ns_ids'], :], axis=1))
        y_shared = batch['r'] + (self.gamma * np.max(Q[batch['ns_ids'], :], axis=1))
        
        # Used mixed model here
        self.q_est_shared.fit(x_shared, y_shared)
        self.q_est_fg.fit(x_fg, y_fg)
        
        return batch_foreground, batch_background
    
    def updateQtable(self, Qtable, batch_fg, batch_bg):
        # Update for foregound using just foreground
        # Update for background using shared
        
        bg_size = len(batch_bg['s'])
        fg_size = len(batch_fg['s'])
        for i, a in enumerate(self.unique_actions):
            Qtable[batch_bg['s_ids'], i] = self.q_est_shared.predict(np.hstack((batch_bg['ns'], np.tile(a, (bg_size,1)))))
            Qtable[batch_fg['s_ids'], i] = self.q_est_fg.predict(np.hstack((batch_fg['ns'], np.tile(a, (fg_size,1)))))
        return Qtable
    
    def runFQI(self, repeats=10):
        
        print('Learning policy')
        meanQtable = np.zeros((self.n_samples + 1, self.n_actions))
        
        for r in range(repeats):
            print('Run', r, ':')
            print('Initialize: get batch, set initial Q')
            Qtable = np.zeros((self.n_samples + 1, self.n_actions))
            Qdist = []

            #print('Run FQI')
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
         
            #plt.plot(Qdist)
            meanQtable += Qtable
        
        meanQtable = meanQtable / repeats
        print('Learn policy')
        
        # Since the Q table is constructed contrastively, the policy is contrastive?
        self.getPi(meanQtable)
        return Qdist
                    
    
    def getPi(self, Qtable):
        optA = np.argmax(Qtable, axis=1)
        rescaled_optA = []
        for a in optA:
            rescaled_optA.append(a - 2)

        optA = np.asarray(rescaled_optA)
        print("Opta: ", optA)
        #print("Fitting to training set")
        #print("Optimal actions: ", optA)
        self.optA = optA[:-1]
        self.piE.fit(self.training_set['s'], optA[:-1])
        #print("Done Fitting")