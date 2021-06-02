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
from sklearn.metrics import (
    log_loss,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
import collections

# import shap
import seaborn as sns
import random
from sklearn.linear_model import LinearRegression, LogisticRegression

np.seterr(all="ignore")
import matplotlib.pyplot as plt
import tqdm
import math
import statsmodels.api as sm
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import sys

sys.path.append("../models/")
from lmm import LMM
import util as util_fqi
from contrastive_deepnet import ContrastiveNet, ContrastiveDataset


class LMMFQIagent:
    def __init__(
        self,
        train_tuples,
        test_tuples,
        iters=150,
        gamma=0.1,
        batch_size=100,
        prioritize=False,
        estimator="gbm",
        weights=np.array([1, 1, 1, 1, 1]) / 5.0,
        maxT=36,
        state_dim=10,
    ):

        self.iters = iters
        self.gamma = gamma
        self.batch_size = batch_size
        self.prioritize_a = prioritize
        self.training_set, self.test_set = util_fqi.construct_dicts(
            train_tuples, test_tuples
        )
        self.raw_test = test_tuples

        self.visits = {"train": len(train_tuples), "test": len(test_tuples)}
        self.NV = {"train": len(train_tuples), "test": len(test_tuples)}
        self.n_samples = len(self.training_set["s"])

        _, self.unique_actions, self.action_counts, _ = self.sub_actions()
        self.state_feats = [str(x) for x in range(10)]
        self.n_features = len(self.state_feats)
        self.reward_weights = weights
        self.maxT = maxT
        self.piB = util_fqi.learnBehaviour(
            self.training_set, self.test_set, state_dim=state_dim
        )
        self.n_actions = len(self.unique_actions)
        print("N actions: ", self.n_actions)
        s = pd.Series(np.arange(self.n_actions))
        self.actions_onehot = pd.get_dummies(s).values
        self.estimator = estimator
        self.state_dim = state_dim

        if self.estimator == "gbm":
            self.q_est = LGBMRegressor(n_estimators=50, silent=True)
        elif self.estimator == "lmm":
            self.q_est = LMM(model="regression", num_classes=self.n_actions)
        elif self.estimator == "linnet":
            self.q_est = ContrastiveNet(model_name="linnet")

        # self.piE = LogisticRegression() #LMM(model='classification', num_classes=self.n_actions)
        self.piE_foreground = LogisticRegression()
        self.piE_background = LogisticRegression()

    def sub_actions(self):

        a = self.training_set["a"]
        a = list(a)

        unique_actions = 0
        action_counts = 0
        n_actions = 0

        unique_actions, action_counts = np.unique(a, axis=0, return_counts=True)
        n_actions = len(unique_actions)

        return a, unique_actions, action_counts, n_actions

    def sampleTuples(self):
        # ids = list(np.random.choice(np.arange(self.n_samples), self.batch_size, replace=False))
        ids = np.arange(self.batch_size)
        batch = {}
        for k in self.training_set.keys():
            batch[k] = np.asarray(self.training_set[k], dtype=object)[ids]
        batch["r"] = batch["r"]  # np.dot(batch['r'], self.reward_weights)

        batch["s_ids"] = np.asarray(ids, dtype=int)
        batch["ns_ids"] = np.asarray(ids, dtype=int) + 1
        return batch

    def fitQ(self, batch, Q):
        if self.estimator == "lmm":
            batch_foreground = {}
            batch_background = {}
            groups = []
            elts = ["s", "s_ids", "ns"]
            for el in elts:
                batch_foreground[el] = []
                batch_background[el] = []

            for i in range(len(batch["s_ids"])):
                if batch["ds"][i] == "foreground":
                    batch_foreground["s_ids"].append(batch["s_ids"][i])
                    batch_foreground["s"].append(batch["s"][i])
                    batch_foreground["ns"].append(batch["ns"][i])
                else:
                    batch_background["s_ids"].append(batch["s_ids"][i])
                    batch_background["s"].append(batch["s"][i])
                    batch_background["ns"].append(batch["ns"][i])

            for i in range(len(batch["s_ids"])):
                if batch["ds"][i] == "foreground":
                    groups.append(1)
                else:
                    groups.append(0)
            # input = [state action]

            s = pd.Series(batch["a"])
            as_onehot = pd.get_dummies(s).values
            x_shared = np.hstack((np.asarray(batch["s"]), as_onehot))
            # x_shared =  np.hstack((np.asarray(batch['s']), np.expand_dims(np.asarray(batch['a']), 1)))

            y_shared = np.squeeze(batch["r"]) + (
                self.gamma * np.max(Q[batch["ns_ids"], :], axis=1)
            )
            self.q_est.fit(x_shared, y_shared, groups)
            return batch, batch_foreground, batch_background

        elif self.estimator == "gbm":
            # input = [state action]
            x = np.hstack(
                (np.asarray(batch["s"]), np.expand_dims(np.asarray(batch["a"]), 1))
            )

            ### Response variable is just the value of the Q table at these indices
            ### 	Important note: at this point we've already acounted for the discount factor gamma
            ###		and we've already taken a max across actions.
            # action_idxs = (batch['a'] + 2).astype(int)
            # Q_states = Q[batch['s_ids'], :]
            # y = Q_states[np.arange(len(Q_states)), action_idxs]

            # old version
            y = np.squeeze(batch["r"]) + (
                self.gamma * np.max(Q[batch["ns_ids"], :], axis=1)
            )
            self.q_est.fit(x, y)

            return batch, None, None

        elif self.estimator == "linnet":
            batch_foreground = {}
            batch_background = {}
            groups = []
            elts = ["s", "s_ids", "ns"]
            for el in elts:
                batch_foreground[el] = []
                batch_background[el] = []

            for i in range(len(batch["s_ids"])):
                if batch["ds"][i] == "foreground":
                    batch_foreground["s_ids"].append(batch["s_ids"][i])
                    batch_foreground["s"].append(batch["s"][i])
                    batch_foreground["ns"].append(batch["ns"][i])
                else:
                    batch_background["s_ids"].append(batch["s_ids"][i])
                    batch_background["s"].append(batch["s"][i])
                    batch_background["ns"].append(batch["ns"][i])

            for i in range(len(batch["s_ids"])):
                if batch["ds"][i] == "foreground":
                    groups.append(1)
                else:
                    groups.append(0)
            ds = ContrastiveDataset(batch, from_batch=True)
            X = ds.X
            y = ds.y
            y = np.squeeze(y) + (self.gamma * np.max(Q[batch["ns_ids"], :], axis=1))
            self.q_est.fit(X, y)
            return None, batch_background, batch_foreground

        else:
            raise Exception(
                "Q_est must be either an LGBM regressor, LMM, or Linear Net"
            )

    def updateQtable(self, Qtable, batch, batch_fg, batch_bg, iter_num):
        # Update for foregound using just foreground
        # Update for background using shared

        if self.estimator == "lmm":
            bg_size = len(batch_bg["s"])
            fg_size = len(batch_fg["s"])

            # for i, a in enumerate(self.unique_actions):
            for i in range(len(self.unique_actions)):
                a = self.actions_onehot[i, :]
                Qtable[batch_bg["s_ids"], i] = self.q_est.predict(
                    np.hstack((batch_bg["ns"], np.tile(a, (bg_size, 1)))),
                    groups=np.tile([0], (bg_size)),
                )
                Qtable[batch_fg["s_ids"], i] = self.q_est.predict(
                    np.hstack((batch_fg["ns"], np.tile(a, (fg_size, 1)))),
                    groups=np.tile([1], (fg_size)),
                )

        elif self.estimator == "gbm":
            for i, a in enumerate(self.unique_actions):
                # import ipdb; ipdb.set_trace()
                Qtable[batch["s_ids"], i] = self.q_est.predict(
                    np.hstack((batch["ns"], np.tile(a, (self.batch_size, 1))))
                )

        elif self.estimator == "linnet":
            for i in range(len(self.unique_actions)):
                for j in range(len(batch_bg["s_ids"])):
                    one_hot_a = [0] * 25
                    s = batch_bg["ns"][j]
                    a = self.unique_actions[i]
                    one_hot_a[a] = 1
                    blank_s = [0] * 46
                    blank_a = [0] * 25
                    s_a = np.hstack((s, one_hot_a, blank_s, blank_a))
                    Qtable[j, i] = self.q_est.predict(s_a.astype("float32"))
            for i in range(len(self.unique_actions)):
                for j in range(len(batch_fg["s_ids"])):
                    one_hot_a = [0] * 25
                    s = batch_fg["ns"][j]
                    a = self.unique_actions[i]
                    one_hot_a[a] = 1
                    s_a = np.hstack((s, one_hot_a, s, one_hot_a))
                    Qtable[j, i] = self.q_est.predict(s_a.astype("float32"))

        return Qtable

    def runFQI(self, repeats=10):

        print("Learning policy")
        meanQtable = np.zeros((self.n_samples + 1, self.n_actions))

        for r in range(repeats):
            print("Run", r, ":")
            print("Initialize: get batch, set initial Q")
            Qtable = np.zeros((self.n_samples + 1, self.n_actions))
            Qdist = []

            # print('Run FQI')
            for k, iteration in enumerate(tqdm.tqdm(range(self.iters))):
                # copy q-table
                Qold = cp.deepcopy(Qtable)

                # sample batch
                batch = self.sampleTuples()

                # learn q_est with samples, targets from batch
                batch, batch_fg, batch_bg = self.fitQ(batch, Qtable)

                # update Q table for all s given new estimator
                self.updateQtable(Qtable, batch, batch_fg, batch_bg, iteration)

                # check divergence from last estimate
                Qdist.append(mean_absolute_error(Qold, Qtable))

            # plt.plot(Qdist)
            meanQtable += Qtable

        meanQtable = meanQtable / repeats
        print("Learn policy")
        self.Qtable = meanQtable
        # Since the Q table is constructed contrastively, the policy is contrastive?
        self.getPi(meanQtable)
        return Qdist

    def getPi(self, Qtable):
        optA = np.argmax(Qtable, axis=1)
        if self.state_dim == 3:
            rescaled_optA = []
            for a in optA:
                rescaled_optA.append(a - 2)

            optA = rescaled_optA

        fg_training = []
        fg_optA = []
        bg_training = []
        bg_optA = []
        for i, g in enumerate(self.training_set["ds"]):
            if g == "foreground":
                fg_training.append(self.training_set["s"][i])
                fg_optA.append(optA[i])
            else:
                bg_training.append(self.training_set["s"][i])
                bg_optA.append(optA[i])

        # import ipdb; ipdb.set_trace()
        # This doesn't on Pendulum env right now because fg_optA is all the same class and bg_optA is all the same class.
        print(str(fg_optA))
        print(str(bg_optA))
        self.piE_foreground.fit(np.asarray(fg_training), fg_optA)
        self.piE_background.fit(np.asarray(bg_training), bg_optA)


# print("Done Fitting")
