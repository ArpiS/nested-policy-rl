import configargparse
import torch
import torch.optim as optim
import sys
sys.path.append('../')

from environments import MountainCarEnv, Continuous_MountainCarEnv
from models.agents import NFQAgent
from models.networks import NFQNetwork, ContrastiveNFQNetwork
from util import get_logger, close_logger, load_models, make_reproducible, save_models
import matplotlib.pyplot as plt
import numpy as np
import itertools
import seaborn as sns
import tqdm

def generate_data(init_experience=400, dataset='train'):
    env_bg = Continuous_MountainCarEnv(group=0)
    env_fg = Continuous_MountainCarEnv(group=1)
    bg_rollouts = []
    fg_rollouts = []
    if init_experience > 0:
        for _ in range(init_experience):
            rollout_bg, episode_cost = env_bg.generate_rollout(
                None, render=False, group=0, dataset=dataset
            )
            rollout_fg, episode_cost = env_fg.generate_rollout(
                None, render=False, group=1, dataset=dataset
            )
            bg_rollouts.extend(rollout_bg)
            fg_rollouts.extend(rollout_fg)
    bg_rollouts.extend(fg_rollouts)
    all_rollouts = bg_rollouts.copy()
    return all_rollouts, env_bg, env_fg

is_contrastive=False
epoch = 1000
evaluations = 10
verbose=True
print("Generating Data")
train_rollouts, train_env_bg, train_env_fg = generate_data(dataset='train')
test_rollouts, eval_env_bg, eval_env_fg = generate_data(dataset='test')

nfq_net = ContrastiveNFQNetwork(
    state_dim=train_env_bg.state_dim, is_contrastive=is_contrastive
)
optimizer = optim.Adam(nfq_net.parameters(), lr=1e-1)

nfq_agent = NFQAgent(nfq_net, optimizer)

# NFQ Main loop
bg_success_queue = [0] * 3
fg_success_queue = [0] * 3
epochs_fg = 0
eval_fg = 0
for k, epoch in enumerate(tqdm.tqdm(range(epoch + 1))):
    state_action_b, target_q_values, groups = nfq_agent.generate_pattern_set(
        train_rollouts
    )
    X = state_action_b
    train_groups = groups

    if not nfq_net.freeze_shared:
        loss = nfq_agent.train((state_action_b, target_q_values, groups))

    eval_episode_length_fg, eval_success_fg, eval_episode_cost_fg = 0, 0, 0
    if nfq_net.freeze_shared:
        eval_fg += 1

        if eval_fg > 50:
            loss = nfq_agent.train((state_action_b, target_q_values, groups))

    (eval_episode_length_bg, eval_success_bg, eval_episode_cost_bg) = nfq_agent.evaluate_car(eval_env_bg, render=False)
    (eval_episode_length_fg,eval_success_fg, eval_episode_cost_fg) = nfq_agent.evaluate_car(eval_env_fg, render=False)

    bg_success_queue = bg_success_queue[1:]
    bg_success_queue.append(1 if eval_success_bg else 0)

    fg_success_queue = fg_success_queue[1:]
    fg_success_queue.append(1 if eval_success_fg else 0)

    printed_bg = False
    printed_fg = False

    if sum(bg_success_queue) == 3 and not nfq_net.freeze_shared == True:
        if epochs_fg == 0:
            epochs_fg = epoch
        printed_bg = True
        nfq_net.freeze_shared = True
        if verbose:
            print("FREEZING SHARED")
        if is_contrastive:
            for param in nfq_net.layers_shared.parameters():
                param.requires_grad = False
            for param in nfq_net.layers_last_shared.parameters():
                param.requires_grad = False
            for param in nfq_net.layers_fg.parameters():
                param.requires_grad = True
            for param in nfq_net.layers_last_fg.parameters():
                param.requires_grad = True
        else:
            for param in nfq_net.layers_fg.parameters():
                param.requires_grad = False
            for param in nfq_net.layers_last_fg.parameters():
                param.requires_grad = False

            optimizer = optim.Adam(
                itertools.chain(
                    nfq_net.layers_fg.parameters(),
                    nfq_net.layers_last_fg.parameters(),
                ),
                lr=1e-1,
            )
            nfq_agent._optimizer = optimizer


    if sum(fg_success_queue) == 3:
        printed_fg = True
        break

eval_env_bg.step_number = 0
eval_env_fg.step_number = 0

eval_env_bg.max_steps = 1000
eval_env_fg.max_steps = 1000

performance_fg = []
performance_bg = []
num_steps_bg = []
num_steps_fg = []
total = 0
for it in range(evaluations):
    (
        eval_episode_length_bg,
        eval_success_bg,
        eval_episode_cost_bg,
    ) = nfq_agent.evaluate_car(eval_env_bg, render=True)
    if verbose:
        print(eval_episode_length_bg, eval_success_bg, eval_episode_cost_bg)
    num_steps_bg.append(eval_episode_length_bg)
    performance_bg.append(eval_episode_length_bg)
    total += 1
    train_env_bg.close()
    eval_env_bg.close()

    (
        eval_episode_length_fg,
        eval_success_fg,
        eval_episode_cost_fg,
    ) = nfq_agent.evaluate_car(eval_env_fg, render=True)
    if verbose:
        print(eval_episode_length_fg, eval_success_fg, eval_episode_cost_fg)
    num_steps_fg.append(eval_episode_length_fg)
    performance_fg.append(eval_episode_length_fg)
    total += 1
    train_env_fg.close()
    eval_env_fg.close()
print("Fg trained after " + str(epochs_fg) + " epochs")
print("BG stayed up for steps: ", num_steps_bg)
print("FG stayed up for steps: ", num_steps_fg)