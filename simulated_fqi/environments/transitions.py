import numpy as np
import pandas as pd
import pickle, os, csv, math, time, joblib
import tqdm
import random
import math


def generate_transitions_LDS():
    # Generate transition matrices, separate distributions for each one
    # We have to ensure that these transitions keep the next state calculations within some reasonable range
    # Make sure that states aren't exploding
    state_dim = 10
    action_dim = 4
    total_dim = state_dim + action_dim

    shape, scale = 2, 10
    transition_foreground = np.random.gamma(shape, scale, (total_dim, state_dim))

    mu, sigma = 0, 4  # mean and standard deviation
    transition_background = np.random.normal(mu, sigma, (total_dim, state_dim))

    # Generate reward function
    mu, sigma = 0, 5
    reward_function = np.random.normal(mu, sigma, (total_dim, 1))

    # Params
    exploit = 0.6
    explore = 1 - exploit
    num_samples = 100
    num_patients = 100
    # actions = [[0, 0], [0, 1], [1, 0], [1, 1]]
    actions = np.eye(4)
    mu, sigma = 0, 4

    transition_tuples = []
    for k, pat in enumerate(tqdm.tqdm(range(num_patients))):

        flip = np.random.choice(2)
        if flip == 0:
            ds = "foreground"
        else:
            ds = "background"
        # Generate a random initial state
        s = np.random.normal(mu, sigma, (state_dim, 1))

        # Generate all of the tuples for this patient
        for i in range(num_samples):
            flip = random.uniform(0, 1)
            # Exploit
            if flip < exploit:
                all_rewards = []
                for j, a in enumerate(actions):
                    a = np.asarray(a)
                    a = np.reshape(a, (action_dim, 1))
                    s_a = np.concatenate((s, a))
                    reward = np.dot(reward_function.T, s_a)
                    all_rewards.append(reward)

                noise = np.random.normal(0, 0.05, 1)
                all_rewards = np.asarray(all_rewards)
                a = actions[np.argmax(all_rewards)]
                reward = np.max(all_rewards) + noise

                if ds == "foreground":
                    t_m = transition_foreground
                else:
                    t_m = transition_background
                ns = np.matmul(s_a.T, t_m) / np.linalg.norm(
                    np.matmul(s_a.T, t_m), ord=2
                )
                ns = np.add(ns, np.random.normal(0, 0.5, (1, state_dim)))  # Add noise

            # Explore
            else:
                a = np.asarray(actions[np.random.choice(action_dim - 1)])
                a = np.reshape(a, (action_dim, 1))
                s_a = np.concatenate((s, a))  # concatenate the state and action

                if ds == "foreground":
                    t_m = transition_foreground
                else:
                    t_m = transition_background
                ns = np.matmul(s_a.T, t_m) / np.linalg.norm(
                    np.matmul(s_a.T, t_m), ord=2
                )
                ns = np.add(ns, np.random.normal(0, 0.5, (1, state_dim)))  # Add noise

                reward = np.dot(reward_function.T, s_a) + np.random.normal(0, 0.5, 1)

            # Transition tuple includes state, action, next state, reward, ds
            transition_tuples.append((s, list(a), ns, reward.flatten(), ds, i))
            s = ns.T

    split = int(0.8 * len(transition_tuples))
    train_tuples = transition_tuples[:split]
    test_tuples = transition_tuples[split:]

    return train_tuples, test_tuples


if __name__ == "__main__":
    pass
