"""Reinforcement learning agents."""
from typing import List, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


class NFQAgent:
    def __init__(self, nfq_net: nn.Module, optimizer: optim.Optimizer):
        """
        Neural Fitted Q-Iteration agent.

        Parameters
        ----------
        nfq_net : nn.Module
            The Q-Network that returns estimated cost given observation and action.
        optimizer : optim.Optimzer
            Optimizer for training the NFQ network.

        """
        self._nfq_net = nfq_net
        self._optimizer = optimizer

    def get_best_action(self, obs: np.array, unique_actions: np.array, group) -> int:
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
        q_list = np.zeros(len(unique_actions))
        for ii, a in enumerate(unique_actions):
            q_list[ii] = self._nfq_net(
                torch.cat([torch.FloatTensor(obs), torch.FloatTensor([a])], dim=0), group * torch.ones(1)
            )

        # plt.plot(q_list)
        # plt.show()

        # Best action has lower "Q" value since it estimates cumulative cost.
        return unique_actions[np.argmin(q_list)]

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
        state_b, action_b, cost_b, next_state_b, done_b, group_b = zip(*rollouts)
        # import ipdb; ipdb.set_trace()
        state_b = torch.FloatTensor(state_b)
        action_b = torch.FloatTensor(action_b)
        cost_b = torch.FloatTensor(cost_b)
        next_state_b = torch.FloatTensor(next_state_b)
        done_b = torch.FloatTensor(done_b)
        group_b = torch.FloatTensor(group_b).unsqueeze(1)

        if len(action_b.size()) == 1:
            action_b = action_b.unsqueeze(1)

        state_action_b = torch.cat([state_b, action_b], 1)
        assert state_action_b.shape == (len(rollouts), state_b.shape[1] + 1)

        # Compute min_a Q(s', a)
        q_next_state_left_b = self._nfq_net(
            torch.cat([next_state_b, torch.zeros(len(rollouts), 1)], 1), group_b
        ).squeeze()
        q_next_state_right_b = self._nfq_net(
            torch.cat([next_state_b, torch.ones(len(rollouts), 1)], 1), group_b
        ).squeeze()

        q_next_state_b = torch.min(q_next_state_left_b, q_next_state_right_b)

        # If goal state (S+): target = 0 + gamma * min Q
        # If forbidden state (S-): target = 1
        # If neither: target = c_trans + gamma * min Q
        # NOTE(seungjaeryanlee): done is True only when the episode terminated
        #                        due to entering forbidden state. It is not
        #                        True if it terminated due to maximum timestep.
        with torch.no_grad():
            target_q_values = cost_b.squeeze() + gamma * q_next_state_b * (1 - done_b)

        return state_action_b, target_q_values, group_b

    def train(self, pattern_set: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """Train neural network with a given pattern set.

        Parameters
        ----------
        pattern_set : tuple of torch.Tensor
            Pattern set to train the NFQ network.

        Returns
        -------
        loss : float
            Training loss.

        """
        state_action_b, target_q_values, groups = pattern_set
        predicted_q_values = self._nfq_net(state_action_b, groups).squeeze()
        loss = F.mse_loss(predicted_q_values, target_q_values)

        # for param in self._nfq_net.parameters():
        #     loss += 10 * torch.norm(param)
        # import ipdb; ipdb.set_trace()
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss.item()

    def evaluate(self, eval_env: gym.Env, render: bool) -> Tuple[int, str, float]:
        """Evaluate NFQ agent on evaluation environment.

        Parameters
        ----------
        eval_env : gym.Env
            Environment to evaluate the agent.
        render: bool
            If true, render environment.

        Returns
        -------
        episode_length : int
            Number of steps the agent took.
        success : bool
            True if the agent was terminated due to max timestep.
        episode_cost : float
            Total cost accumulated from the evaluation episode.

        """
        episode_length = 0
        obs = eval_env.reset()
        done = False
        info = {"time_limit": False}
        episode_cost = 0
        while not done and not info["time_limit"]:
            action = self.get_best_action(obs, eval_env.unique_actions, eval_env.group)
            obs, cost, done, info = eval_env.step(action)
            episode_cost += cost
            episode_length += 1

            if render:
                eval_env.render()


        success = (
            episode_length == eval_env.max_steps
            and abs(obs[0]) <= eval_env.x_success_range
        )

        return episode_length, success, episode_cost


    def evaluate_cart(self, eval_env: gym.Env, render: bool) -> Tuple[int, str, float]:
        """Evaluate NFQ agent on evaluation environment.

        Parameters
        ----------
        eval_env : gym.Env
            Environment to evaluate the agent.
        render: bool
            If true, render environment.

        Returns
        -------
        episode_length : int
            Number of steps the agent took.
        success : bool
            True if the agent was terminated due to max timestep.
        episode_cost : float
            Total cost accumulated from the evaluation episode.

        """
        success_length = 0
        obs = eval_env.reset()
        done = False
        info = {"time_limit": False}
        episode_cost = 0
        while not done and not info["time_limit"]:
            action = self.get_best_action(obs, eval_env.unique_actions, eval_env.group)
            obs, cost, done, info = eval_env.step(action)
            episode_cost += cost
            if cost == 0:
                success_length += 1

            if render:
                eval_env.render()


        success = (
            success_length == eval_env.max_steps
            and abs(obs[0]) <= eval_env.x_success_range
        )

        return success_length, success, episode_cost

    def evaluate_pendulum(self, eval_env: gym.Env, num_steps: int = 100, render: bool = False) -> Tuple[int, str, float]:
        episode_length = 0
        obs = eval_env.reset()
        done = False
        info = {"time_limit": False}
        episode_cost = 0
        for ii in range(num_steps):
            action = self.get_best_action(obs, eval_env.unique_actions, eval_env.group)
            obs, cost, done, info = eval_env.step(action)
            episode_cost += cost
            episode_length += 1

            if render:
                eval_env.render()

        success = (
            episode_cost / num_steps * 1.0 == -1
        )

        return episode_length, success, episode_cost
