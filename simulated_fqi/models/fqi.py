import numpy as np
import pandas as pd
import pickle, os, csv, math, time, joblib
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression, LogisticRegression
# import util as util_fqi
import util
from util import util as util_fqi
import copy as cp
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.getcwd()))+"/environments")
from cartpole import CartPoleRegulatorEnv


class FQIagent:
    def __init__(
        self,
        train_tuples,
        test_tuples,
        iters=150,
        gamma=0.99,
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
        self.state_dim = state_dim
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
        # self.piB = util_fqi.learnBehaviour(self.training_set, self.test_set, state_dim=state_dim)
        self.n_actions = len(self.unique_actions)
        print("N actions: ", self.n_actions)

        if estimator == "tree":
            self.q_est = ExtraTreesRegressor(
                n_estimators=50,
                max_depth=None,
                min_samples_leaf=10,
                min_samples_split=2,
                random_state=0,
            )
        elif estimator == "gbm":
            self.q_est = LGBMRegressor(n_estimators=50, silent=True)

        elif estimator == "nn":
            self.q_est = None

        elif estimator == "lin":
            self.q_est = LinearRegression()

        self.piE = LogisticRegression()  # LinearRegression()

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

        # # Get a batch of unprioritized samples:
        #
        # ids = list(np.random.choice(np.arange(self.n_samples), self.batch_size, replace=False))
        # batch = {}
        # for k in self.training_set.keys():
        #     batch[k] = np.asarray(self.training_set[k], dtype=object)[ids]
        # batch['r'] = np.dot(batch['r'] * [1, 1, 10, 10, 100], self.reward_weights)
        # batch['s_ids'] = np.asarray(ids, dtype=int)
        # batch['ns_ids'] = np.asarray(ids, dtype=int) + 1
        #
        #
        # return batch
        ids = list(
            np.random.choice(np.arange(self.n_samples), self.batch_size, replace=False)
        )
        batch = {}
        for k in self.training_set.keys():
            batch[k] = np.asarray(self.training_set[k], dtype=object)[ids]
        batch["r"] = batch["r"]  # np.dot(batch['r'], self.reward_weights)

        batch["s_ids"] = np.asarray(ids, dtype=int)
        batch["ns_ids"] = np.asarray(ids, dtype=int) + 1

        return batch

    def fitQ(self, batch, Q):

        # input = [state action]
#         print(np.asarray(batch["a"]).shape)
#         print(np.asarray(batch["s"]))
#         print(np.expand_dims(np.asarray(batch["a"]), 1))
#         print(np.asarray(batch["a"]))
        x = np.hstack(
            (np.asarray(batch["s"]), np.asarray(batch["a"]))
#             (np.asarray(batch["s"]), np.expand_dims(np.asarray(batch["a"]), 1))
        )

        # target = r + gamma * max_a(Q(s', a))      == r for first iteration
        y = np.squeeze(batch["r"]) + (
            self.gamma * np.max(Q[batch["ns_ids"], :], axis=1)
        )

        self.q_est.fit(x, y)

    def updateQtable(self, Qtable, batch):

        for i, a in enumerate(self.unique_actions):
            Qtable[batch["s_ids"], i] = self.q_est.predict(
                np.hstack((batch["ns"], np.tile(a, (self.batch_size, 1))))
            )
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
            for iteration in range(self.iters):

                # copy q-table
                Qold = cp.deepcopy(Qtable)

                # sample batch
                batch = self.sampleTuples()

                # learn q_est with samples, targets from batch
                self.fitQ(batch, Qtable)

                # update Q table for all s given new estimator
                self.updateQtable(Qtable, batch)

                # check divergence from last estimate
                Qdist.append(mean_absolute_error(Qold, Qtable))

            meanQtable += Qtable

        meanQtable = meanQtable / repeats

        self.Qtable = meanQtable
        print("Learn policy")
        self.getPi(meanQtable)
        return Qdist

    def sampleTuplesTorch(self):
        tuples = self.sampleTuples()

        state_b = tuples["s"].astype("float")
        action_b = tuples["a"].astype("float")
        cost_b = tuples["r"].astype("float")
        next_state_b = tuples["ns"].astype("float")
        n_steps = len(tuples["a"])
        # , cost_b, next_state_b, done_b = zip(*tuples)
        # import ipdb; ipdb.set_trace()
        state_b = torch.FloatTensor(state_b)

        action_b = torch.FloatTensor(action_b)
        cost_b = torch.FloatTensor(cost_b)
        next_state_b = torch.FloatTensor(next_state_b)
        # done_b = torch.FloatTensor(done_b)

        state_action_b = torch.cat([state_b, action_b.unsqueeze(1)], 1)
        assert state_action_b.shape == (n_steps, state_b.shape[1] + 1)

        # Compute min_a Q(s', a)
        q_next_state_left_b = self._nfq_net(
            torch.cat([next_state_b, torch.zeros(n_steps, 1)], 1)
        ).squeeze()
        q_next_state_right_b = self._nfq_net(
            torch.cat([next_state_b, torch.ones(n_steps, 1)], 1)
        ).squeeze()
        q_next_state_b = torch.min(q_next_state_left_b, q_next_state_right_b)

        # If goal state (S+): target = 0 + gamma * min Q
        # If forbidden state (S-): target = 1
        # If neither: target = c_trans + gamma * min Q
        # NOTE(seungjaeryanlee): done is True only when the episode terminated
        #                        due to entering forbidden state. It is not
        #                        True if it terminated due to maximum timestep.
        with torch.no_grad():
            target_q_values = (
                cost_b.squeeze() + self.gamma * q_next_state_b
            )  # * (1 - done_b)
        # import ipdb; ipdb.set_trace()

        return state_action_b, target_q_values

    def generate_pattern_sets(self, rollouts):
        state_b, action_b, cost_b, next_state_b, done_b = zip(*rollouts)
        state_b = torch.FloatTensor(state_b)
        action_b = torch.FloatTensor(action_b)
        cost_b = torch.FloatTensor(cost_b)
        next_state_b = torch.FloatTensor(next_state_b)
        done_b = torch.FloatTensor(done_b)

        state_action_b = torch.cat([state_b, action_b.unsqueeze(1)], 1)
        assert state_action_b.shape == (len(rollouts), state_b.shape[1] + 1)

        # Compute min_a Q(s', a)
        q_next_state_left_b = self._nfq_net(
            torch.cat([next_state_b, torch.zeros(len(rollouts), 1)], 1)
        ).squeeze()
        q_next_state_right_b = self._nfq_net(
            torch.cat([next_state_b, torch.ones(len(rollouts), 1)], 1)
        ).squeeze()
        q_next_state_b = torch.min(q_next_state_left_b, q_next_state_right_b)

        with torch.no_grad():
            target_q_values = cost_b + self.gamma * q_next_state_b * (1 - done_b)

        return state_action_b, target_q_values

    def runNFQI(self):
        class NFQNetwork(nn.Module):
            def __init__(self, state_dim):
                """Networks for NFQ."""
                super().__init__()

                self.layers = nn.Sequential(
                    nn.Linear(state_dim + 1, 5),
                    nn.Sigmoid(),
                    nn.Linear(5, 5),
                    nn.Sigmoid(),
                    nn.Linear(5, 1),
                    nn.Sigmoid(),
                )

                def init_weights(m):
                    if type(m) == nn.Linear:
                        torch.nn.init.uniform_(m.weight, -0.5, 0.5)

                self.layers.apply(init_weights)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.layers(x)

        # Setup agent
        nfq_net = NFQNetwork(self.state_dim)
        self._nfq_net = nfq_net
        self.optimizer = optim.Rprop(nfq_net.parameters())

        train_env = CartPoleRegulatorEnv()

        all_rollouts = []
        for _ in range(200):
            rollout, episode_cost = train_env.generate_rollout(None, render=False)
            all_rollouts.extend(rollout)

        loss_trace = []
        for ii in range(self.iters):
            # state_action_b, target_q_values = self.sampleTuplesTorch()

            state_action_b, target_q_values = self.generate_pattern_sets(all_rollouts)

            curr_loss = self.trainNFQI(state_action_b, target_q_values)

            loss_trace.append(curr_loss)

            episode_length, success, episode_cost = self.evaluateNFQI()
            print(
                "Loss: {}, Eval length: {}".format(round(curr_loss, 3), episode_length)
            )

        plt.plot(loss_trace)
        plt.show()

    def trainNFQI(self, states, targets):
        predicted_q_values = self._nfq_net(states).squeeze()
        loss = F.mse_loss(predicted_q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predictNFQI(self, obs: np.array) -> int:
        actions = np.sort(self.sub_actions()[1])
        q_list = np.zeros(len(actions))
        for ii, a in enumerate(actions):
            curr_q_est = self._nfq_net(
                torch.cat([torch.FloatTensor(obs), torch.FloatTensor([a])], dim=0)
            )
            q_list[ii] = curr_q_est
        return np.argmin(q_list)
        # return 1 if q_list[0] >= q_list[1] else 0

    def evaluateNFQI(self, render=False):
        eval_env = CartPoleRegulatorEnv()
        episode_length = 0
        obs = eval_env.reset()
        done = False
        time_limit = 2000
        episode_cost = 0
        while not done:
            action = self.predictNFQI(obs)
            curr_state, cost, done, _ = eval_env.step(action)
            episode_cost += cost
            episode_length += 1

            if render:
                eval_env.render()

            if episode_length == time_limit:
                break

        success = (
            episode_length == eval_env.max_steps
            and abs(obs[0]) <= eval_env.x_success_range
        )

        return episode_length, success, episode_cost

    def getPi(self, Qtable):
        unique_counts = (Qtable[:, 1:] != Qtable[:, :-1]).sum(axis=1) + 1
        uniform_rows = np.where(unique_counts == 1)[0]
        optA = np.argmax(Qtable, axis=1)
        optA[uniform_rows] = np.random.choice(
            [0, 1], size=len(uniform_rows), replace=True
        )

        if self.state_dim == 3:
            rescaled_optA = []
            for a in optA:
                rescaled_optA.append(a - 2)

            optA = np.asarray(rescaled_optA)
        self.optA = optA[:-1]
        self.piE.fit(self.training_set["s"], optA[:-1])
        print("Fit score: ", self.piE.score(self.training_set["s"], optA[:-1]))
