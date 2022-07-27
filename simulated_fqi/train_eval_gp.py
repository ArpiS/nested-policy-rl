import configargparse
import torch
import torch.optim as optim

from environments import CartPoleRegulatorEnv
from environments import CartEnv
from environments import AcrobotEnv
from models.agents import NFQAgent
from models.networks import NFQNetwork, ContrastiveNFQNetwork
from util import get_logger, close_logger, load_models, make_reproducible, save_models
import matplotlib.pyplot as plt
import numpy as np
import itertools
import sys


"""
Runs CFQI and standard FQI.

Parameters
----------
verbose : bool
    A flag used to print progress to the console. 
is_contrastive: bool
    A flag used to determine if the algorithm should be contrastive. When True, runs CFQI. When False, runs FQI
epoch: int
    Max number of epochs to train the model. 
init_experience: int
    Number of training trajectories for foreground and background. 
evaluations: int
    Number of times to evaluate the model.
force_left: int
    Force acting on the foreground environment. A higher number makes the background and foreground dataset more different. 
random_seed: int
    Random seed to ensure reproducibility. 

Returns
-------
nfq_agent
    Training agent, can access the model using nfq_agent._nfq_net.
"""


def fqi(
    verbose=True,
    is_contrastive=False,
    epoch=1000,
    init_experience=200,
    evaluations=5,
    force_left=5,
    # random_seed=None,
    random_seed=7,
):

    # Setup environment
    bg_cart_mass = 1.0
    fg_cart_mass = 1.0
    train_env_bg = CartPoleRegulatorEnv(
        group=0,
        masscart=bg_cart_mass,
        mode="train",
        force_left=force_left,
        is_contrastive=is_contrastive,
    )
    train_env_fg = CartPoleRegulatorEnv(
        group=1,
        masscart=fg_cart_mass,
        mode="train",
        force_left=force_left,
        is_contrastive=is_contrastive,
    )
    eval_env_bg = CartPoleRegulatorEnv(
        group=0,
        masscart=bg_cart_mass,
        mode="eval",
        force_left=force_left,
        is_contrastive=is_contrastive,
    )
    eval_env_fg = CartPoleRegulatorEnv(
        group=1,
        masscart=fg_cart_mass,
        mode="eval",
        force_left=force_left,
        is_contrastive=is_contrastive,
    )

    if random_seed is not None:
        make_reproducible(random_seed, use_numpy=True, use_torch=True)
        train_env_bg.seed(random_seed)
        train_env_fg.seed(random_seed)
        eval_env_bg.seed(random_seed)
        eval_env_fg.seed(random_seed)

    # Log to File, Console, TensorBoard, W&B
    logger = get_logger()

    # Setup agent
    nfq_net = ContrastiveNFQNetwork(
        state_dim=train_env_bg.state_dim, is_contrastive=is_contrastive
    )

    if is_contrastive:
        optimizer = optim.Adam(
            itertools.chain(
                nfq_net.layers_shared.parameters(),
                nfq_net.layers_last_shared.parameters(),
            ),
            lr=1e-1,
        )
    else:
        optimizer = optim.Adam(nfq_net.parameters(), lr=1e-1)

    nfq_agent = NFQAgent(nfq_net, optimizer)

    # NFQ Main loop
    bg_rollouts = []
    fg_rollouts = []
    total_cost = 0
    if init_experience > 0:
        for _ in range(init_experience):
            rollout_bg, episode_cost = train_env_bg.generate_rollout(
                None, render=False, group=0
            )
            rollout_fg, episode_cost = train_env_fg.generate_rollout(
                None, render=False, group=1
            )
            bg_rollouts.extend(rollout_bg)
            fg_rollouts.extend(rollout_fg)
            total_cost += episode_cost
    only_bg_rollouts = bg_rollouts.copy()
    bg_rollouts.extend(fg_rollouts)
    all_rollouts = bg_rollouts.copy()

    bg_rollouts_test = []
    fg_rollouts_test = []
    if init_experience > 0:
        for _ in range(init_experience):
            rollout_bg, episode_cost = eval_env_bg.generate_rollout(
                None, render=False, group=0
            )
            rollout_fg, episode_cost = eval_env_fg.generate_rollout(
                None, render=False, group=1
            )
            bg_rollouts_test.extend(rollout_bg)
            fg_rollouts_test.extend(rollout_fg)
    only_bg_rollouts_test = bg_rollouts_test.copy()
    bg_rollouts_test.extend(fg_rollouts)
    all_rollouts_test = bg_rollouts_test.copy()

    x_fg_train, reward_fg_train = nfq_agent.generate_state_action_rewards(fg_rollouts)
    x_bg_train, reward_bg_train = nfq_agent.generate_state_action_rewards(only_bg_rollouts)

    x_fg_test, reward_fg_test = nfq_agent.generate_state_action_rewards(fg_rollouts_test)
    x_bg_test, reward_bg_test = nfq_agent.generate_state_action_rewards(only_bg_rollouts_test)

    state_action_b, target_q_values, groups = nfq_agent.generate_pattern_set(
        all_rollouts_test
    )
    # X_test = state_action_b
    test_groups = groups

    bg_success_queue = [0] * 3
    fg_success_queue = [0] * 3
    epochs_fg = 0
    eval_fg = 0
    # for epoch in range(epoch + 1):

    #     state_action_b, target_q_values, groups = nfq_agent.generate_pattern_set(
    #         all_rollouts
    #     )
    #     X = state_action_b
    #     train_groups = groups

    #     if not nfq_net.freeze_shared:
    #         loss = nfq_agent.train((state_action_b, target_q_values, groups))

    #     eval_episode_length_fg, eval_success_fg, eval_episode_cost_fg = 0, 0, 0
    #     if nfq_net.freeze_shared:
    #         eval_fg += 1

    #         if eval_fg > 50:
    #             loss = nfq_agent.train((state_action_b, target_q_values, groups))

    #     if is_contrastive:
    #         if nfq_net.freeze_shared:
    #             (
    #                 eval_episode_length_fg,
    #                 eval_success_fg,
    #                 eval_episode_cost_fg,
    #             ) = nfq_agent.evaluate(eval_env_fg, render=False, is_mggp=True)
    #             for param in nfq_net.layers_fg.parameters():
    #                 assert param.requires_grad == True
    #             for param in nfq_net.layers_last_fg.parameters():
    #                 assert param.requires_grad == True
    #             for param in nfq_net.layers_shared.parameters():
    #                 assert param.requires_grad == False
    #             for param in nfq_net.layers_last_shared.parameters():
    #                 assert param.requires_grad == False
    #         else:
    #             for param in nfq_net.layers_fg.parameters():
    #                 assert param.requires_grad == False
    #             for param in nfq_net.layers_last_fg.parameters():
    #                 assert param.requires_grad == False
    #             for param in nfq_net.layers_shared.parameters():
    #                 assert param.requires_grad == True
    #             for param in nfq_net.layers_last_shared.parameters():
    #                 assert param.requires_grad == True
    #             (
    #                 eval_episode_length_bg,
    #                 eval_success_bg,
    #                 eval_episode_cost_bg,
    #             ) = nfq_agent.evaluate(eval_env_bg, render=False, is_mggp=True)

    #     else:
    #         (
    #             eval_episode_length_bg,
    #             eval_success_bg,
    #             eval_episode_cost_bg,
    #         ) = nfq_agent.evaluate(eval_env_bg, render=False, is_mggp=True)
    #         (
    #             eval_episode_length_fg,
    #             eval_success_fg,
    #             eval_episode_cost_fg,
    #         ) = nfq_agent.evaluate(eval_env_fg, render=False, is_mggp=True)

    #     bg_success_queue = bg_success_queue[1:]
    #     bg_success_queue.append(1 if eval_success_bg else 0)

    #     fg_success_queue = fg_success_queue[1:]
    #     fg_success_queue.append(1 if eval_success_fg else 0)

    #     printed_bg = False
    #     printed_fg = False

    #     if sum(bg_success_queue) == 3 and not nfq_net.freeze_shared == True:
    #         if epochs_fg == 0:
    #             epochs_fg = epoch
    #         printed_bg = True
    #         nfq_net.freeze_shared = True
    #         if verbose:
    #             print("FREEZING SHARED")
    #         if is_contrastive:
    #             for param in nfq_net.layers_shared.parameters():
    #                 param.requires_grad = False
    #             for param in nfq_net.layers_last_shared.parameters():
    #                 param.requires_grad = False
    #             for param in nfq_net.layers_fg.parameters():
    #                 param.requires_grad = True
    #             for param in nfq_net.layers_last_fg.parameters():
    #                 param.requires_grad = True
    #         else:
    #             for param in nfq_net.layers_fg.parameters():
    #                 param.requires_grad = False
    #             for param in nfq_net.layers_last_fg.parameters():
    #                 param.requires_grad = False

    #             optimizer = optim.Adam(
    #                 itertools.chain(
    #                     nfq_net.layers_fg.parameters(),
    #                     nfq_net.layers_last_fg.parameters(),
    #                 ),
    #                 lr=1e-1,
    #             )
    #             nfq_agent._optimizer = optimizer
    #         # break

    #     # Print current status
    #     if verbose:
    #         logger.info(
    #             "Epoch {:4d} | Eval BG {:4d} / {:4f} | Eval FG {:4d} / {:4f} | Train Loss {:.4f}".format(
    #                 epoch,
    #                 eval_episode_length_bg,
    #                 eval_episode_cost_bg,
    #                 eval_episode_length_fg,
    #                 eval_episode_cost_fg,
    #                 loss,
    #             )
    #         )
    #     if sum(fg_success_queue) == 3:
    #         printed_fg = True
    #         if verbose:
    #             logger.info(
    #                 "Epoch {:4d} | Total Cycles {:6d} | Total Cost {:4.2f}".format(
    #                     epoch, len(all_rollouts), total_cost
    #                 )
    #             )
    #         break

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
        ) = nfq_agent.evaluate(eval_env_bg, render=False, is_mggp=True)
        if verbose:
            print(eval_episode_length_bg, eval_success_bg)
        num_steps_bg.append(eval_episode_length_bg)
        performance_bg.append(eval_episode_length_bg)
        total += 1
        train_env_bg.close()
        eval_env_bg.close()

        (
            eval_episode_length_fg,
            eval_success_fg,
            eval_episode_cost_fg,
        ) = nfq_agent.evaluate(eval_env_fg, render=False, is_mggp=True)
        if verbose:
            print(eval_episode_length_fg, eval_success_fg)
        num_steps_fg.append(eval_episode_length_fg)
        performance_fg.append(eval_episode_length_fg)
        total += 1
        train_env_fg.close()
        eval_env_fg.close()
    # print("Fg trained after " + str(epochs_fg) + " epochs")
    print("BG stayed up for steps: ", num_steps_bg)
    print("FG stayed up for steps: ", num_steps_fg)

    state_action_b, target_q_values, groups = nfq_agent.generate_pattern_set(
        all_rollouts
    )
    X = (state_action_b, groups)

    bg_rollouts = []
    fg_rollouts = []
    total_cost = 0
    init_experience = 50
    if init_experience > 0:
        for _ in range(init_experience):
            rollout_bg, episode_cost = eval_env_bg.generate_rollout(
                None, render=False, group=0
            )
            rollout_fg, episode_cost = eval_env_fg.generate_rollout(
                None, render=False, group=1
            )
            bg_rollouts.extend(rollout_bg)
            fg_rollouts.extend(rollout_fg)

    bg_rollouts.extend(fg_rollouts)
    all_rollouts = bg_rollouts.copy()
    state_action_b, target_q_values, groups = nfq_agent.generate_pattern_set(
        all_rollouts
    )
    X_test = (state_action_b, groups)

    return nfq_agent, X_test, (x_fg_train, reward_fg_train, x_bg_train, reward_bg_train), (x_fg_test, reward_fg_test, x_bg_test, reward_bg_test)


def save_matrices(x_fg, reward_fg, x_bg, reward_bg, suffix):
    torch.save(x_fg, "matrices/x_fg_"+suffix)
    torch.save(reward_fg, "matrices/reward_fg_"+suffix)
    torch.save(x_bg, "matrices/x_bg_"+suffix)
    torch.save(reward_bg, "matrices/reward_bg_"+suffix)


if __name__ == "__main__":
    algorithm = sys.argv[1]

    if algorithm == "fqi":
        nfq_agent, X_test = fqi(is_contrastive=False)
        torch.save(nfq_agent._nfq_net, "checkpoints/fqi_model")
    elif algorithm == "cfqi":
        nfq_agent, X_test, train, test = fqi(is_contrastive=True)
        torch.save(nfq_agent._nfq_net, "checkpoints/cfqi_model")
        save_matrices(*train, "train")
        save_matrices(*test, "test")
    
    else:
        print("Invalid argument. Options are fqi, cfqi.")
