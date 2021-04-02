import numpy as np
import pandas as pd
import pickle, os, csv, math, time, joblib
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression, LogisticRegression
import util as util_fqi
import copy as cp
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import sys
from nfq_network import NFQNetwork
from typing import List, Tuple
sys.path.append("../environments")
from cartpole import CartPoleRegulatorEnv

class NFQIagent():
    def __init__(self, unique_actions, iters=150, gamma=0.99, state_dim=10):
        
        self.iters = iters
        self.gamma = gamma
        self.state_dim = state_dim
        
        self.unique_actions = unique_actions
        self.n_actions = len(self.unique_actions)

        # Setup agent
        self._nfq_net = NFQNetwork(self.state_dim)
        self.optimizer = optim.Rprop(self._nfq_net.parameters())

        # Set up environments
        self.train_env = CartPoleRegulatorEnv(mode="train")
        self.eval_env = CartPoleRegulatorEnv(mode="eval")
        print("N actions: ", self.n_actions)

    def sub_actions(self):
        
        a = self.training_set['a']
        a = list(a)
        
        unique_actions = 0
        action_counts = 0
        n_actions = 0
        
        unique_actions, action_counts = np.unique(a, axis=0, return_counts=True)
        n_actions = len(unique_actions)
                
        return a, unique_actions, action_counts, n_actions

    def generate_pattern_set(
            self,
            rollouts: List[Tuple[np.array, int, int, np.array, bool]],
            gamma: float = 0.95,
        ):
            """Generate pattern set.
            Parameters
            ----------
            rollouts : list of tuple
                Generated rollouts, which is a tuple of state, action, cost, next state, and done.
            gamma : float
                Discount factor. Defaults to 0.95.
            Returns
            -------
            pattern_set : tuple of torch.Tensor
                Pattern set to train the NFQ network.
            """
            # _b denotes batch
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

            # If goal state (S+): target = 0 + gamma * min Q
            # If forbidden state (S-): target = 1
            # If neither: target = c_trans + gamma * min Q
            # NOTE(seungjaeryanlee): done is True only when the episode terminated
            #                        due to entering forbidden state. It is not
            #                        True if it terminated due to maximum timestep.
            with torch.no_grad():
                target_q_values = cost_b + gamma * q_next_state_b * (1 - done_b)

            return state_action_b, target_q_values

    def runNFQI(self):

        # Generate data
        all_rollouts = []
        for _ in range(200):
            rollout, episode_cost = self.train_env.generate_rollout(
                None, render=False
            )
            all_rollouts.extend(rollout)

        loss_trace = []
        for ii in range(self.iters):
            state_action_b, target_q_values = self.generate_pattern_set(all_rollouts, gamma=self.gamma)

            curr_loss = self.trainNFQI(state_action_b, target_q_values)

            loss_trace.append(curr_loss)

            episode_length, success, episode_cost = self.evaluateNFQI(self.eval_env)
            print("Loss: {}, Eval length: {}".format(round(curr_loss, 3), episode_length))

        plt.plot(loss_trace)
        plt.show()
        self.evaluateNFQI(self.eval_env, True)

    def trainNFQI(self, states, targets):
        predicted_q_values = self._nfq_net(states).squeeze()
        loss = F.mse_loss(predicted_q_values, targets)
        # print(predicted_q_values)
        # import ipdb; ipdb.set_trace()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_best_action(self, obs: np.array) -> int:
        """
        Return best action for given observation according to the neural network.
        Parameters
        ----------
        obs : np.array
            An observation to find the best action for.
        Returns
        -------
        action : int
            The action chosen by greedy selection.
        """
        q_left = self._nfq_net(
            torch.cat([torch.FloatTensor(obs), torch.FloatTensor([0])], dim=0)
        )
        q_right = self._nfq_net(
            torch.cat([torch.FloatTensor(obs), torch.FloatTensor([1])], dim=0)
        )

        # Best action has lower "Q" value since it estimates cumulative cost.
        return 1 if q_left >= q_right else 0

    # def predictNFQI(self, obs: np.array) -> int:
    #     q_list = np.zeros(len(self.unique_actions))
    #     for ii, a in enumerate(self.unique_actions):
    #         curr_q_est = self._nfq_net(
    #             torch.cat([torch.FloatTensor(obs), torch.FloatTensor([a])], dim=0)
    #         )
    #         q_list[ii] = curr_q_est
    #     return np.argmin(q_list)
        # return 1 if q_list[0] >= q_list[1] else 0

    def evaluateNFQI(self, eval_env, render=False):
        episode_length = 0
        obs = eval_env.reset()
        done = False
        time_limit = 2000
        episode_cost = 0
        while not done:
            action = self.get_best_action(obs)
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


