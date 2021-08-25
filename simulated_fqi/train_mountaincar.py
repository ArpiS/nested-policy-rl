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

def generate_data(init_experience=100, bg_only=False, continuous=False, agent=None, dataset='train'):
    if continuous:
        env_bg = Continuous_MountainCarEnv(group=0)
        env_fg = Continuous_MountainCarEnv(group=1)
    else:
        env_bg = MountainCarEnv(group=0)
        env_fg = MountainCarEnv(group=1)
    bg_rollouts = []
    fg_rollouts = []
    if init_experience > 0:
        for _ in range(init_experience):
            rollout_bg, episode_cost = env_bg.generate_rollout(
                agent, render=False, group=0, dataset=dataset
            )
            bg_rollouts.extend(rollout_bg)
            if not bg_only:
                rollout_fg, episode_cost = env_fg.generate_rollout(
                    agent, render=False, group=1, dataset=dataset
                )
                fg_rollouts.extend(rollout_fg)
    bg_rollouts.extend(fg_rollouts)
    all_rollouts = bg_rollouts.copy()
    return all_rollouts, env_bg, env_fg

def fqi(is_contrastive, epoch, hint_to_goal=True, init_experience=200, verbose=False):
    train_rollouts, train_env_bg, train_env_fg = generate_data(init_experience=init_experience, bg_only=False, continuous=False)
    test_rollouts, eval_env_bg, eval_env_fg = generate_data(init_experience=init_experience, bg_only=False, continuous=False)

    if hint_to_goal:
        goal_state_action_b_bg, goal_target_q_values_bg, group_bg = train_env_bg.get_goal_pattern_set(group=0)
        goal_state_action_b_fg, goal_target_q_values_fg, group_fg = train_env_fg.get_goal_pattern_set(group=1)

        goal_state_action_b_bg = torch.FloatTensor(goal_state_action_b_bg)
        goal_target_q_values_bg = torch.FloatTensor(goal_target_q_values_bg)
        goal_state_action_b_fg = torch.FloatTensor(goal_state_action_b_fg)
        goal_target_q_values_fg = torch.FloatTensor(goal_target_q_values_fg)

    nfq_net = ContrastiveNFQNetwork(state_dim=train_env_bg.state_dim, is_contrastive=is_contrastive, deep=False)
    optimizer = optim.Adam(nfq_net.parameters(), lr=1e-1)

    nfq_agent = NFQAgent(nfq_net, optimizer)

    bg_success_queue = [0] * 3
    fg_success_queue = [0] * 3
    eval_fg = 0
    evaluations = 5
    for k, ep in enumerate(tqdm.tqdm(range(epoch + 1))):
        state_action_b, target_q_values, groups = nfq_agent.generate_pattern_set(train_rollouts)
        if hint_to_goal:
            goal_state_action_b = torch.cat([goal_state_action_b_bg, goal_state_action_b_fg], dim=0)
            goal_target_q_values = torch.cat([goal_target_q_values_bg, goal_target_q_values_fg], dim=0)
            state_action_b = torch.cat([state_action_b, goal_state_action_b], dim=0)
            target_q_values = torch.cat([target_q_values, goal_target_q_values], dim=0)
            goal_groups = torch.cat([group_bg, group_fg], dim=0)
            groups = torch.cat([groups, goal_groups], dim=0)

        if not nfq_net.freeze_shared:
            loss = nfq_agent.train((state_action_b, target_q_values, groups))

        eval_episode_length_fg, eval_success_fg, eval_episode_cost_fg = 0, 0, 0
        if nfq_net.freeze_shared:
            eval_fg += 1
            if eval_fg > 50:
                loss = nfq_agent.train((state_action_b, target_q_values, groups))

        (eval_episode_length_bg, eval_success_bg, eval_episode_cost_bg) = nfq_agent.evaluate_car(eval_env_bg, render=False)
        bg_success_queue = bg_success_queue[1:]
        bg_success_queue.append(1 if eval_success_bg else 0)

        (eval_episode_length_fg, eval_success_fg, eval_episode_cost_fg) = nfq_agent.evaluate_car(eval_env_fg, render=False)
        fg_success_queue = fg_success_queue[1:]
        fg_success_queue.append(1 if eval_success_fg else 0)

        if sum(bg_success_queue) == 3 and not nfq_net.freeze_shared == True:
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
            if verbose:
                print("FG Trained")
            break

        if ep % 300 == 0:
            perf_bg = []
            perf_fg = []
            for it in range(evaluations):
                (eval_episode_length_bg, eval_success_bg, eval_episode_cost_bg) = nfq_agent.evaluate_car(eval_env_bg, render=False)
                (eval_episode_length_fg, eval_success_fg, eval_episode_cost_fg) = nfq_agent.evaluate_car(eval_env_fg, render=False)
                perf_bg.append(eval_episode_cost_bg)
                perf_fg.append(eval_episode_cost_fg)
                train_env_bg.close()
                train_env_fg.close()
                eval_env_bg.close()
                eval_env_fg.close()
            if verbose:
                print("Evaluation bg: " + str(perf_bg) + " Evaluation fg: " + str(perf_fg))
    perf_bg = []
    perf_fg = []
    for it in range(evaluations * 10):
        (eval_episode_length_bg, eval_success_bg, eval_episode_cost_bg) = nfq_agent.evaluate_car(eval_env_bg, render=False)
        (eval_episode_length_fg, eval_success_fg, eval_episode_cost_fg) = nfq_agent.evaluate_car(eval_env_fg, render=False)
        perf_bg.append(eval_episode_cost_bg)
        perf_fg.append(eval_episode_cost_fg)
        eval_env_bg.close()
        eval_env_fg.close()
    if verbose:
        print("Evaluation bg: " + str(sum(perf_bg) / len(perf_bg)) + " Evaluation fg: " + str(sum(perf_fg) / len(perf_fg)))
    return sum(perf_bg)/len(perf_bg), sum(perf_fg)/len(perf_fg)

def warm_start(
    verbose=True,
    epoch=1000,
    init_experience=200,
    evaluations=5,
    force_left=5,
    random_seed=None,
):
    # Setup environment
    bg_cart_mass = 1.0
    fg_cart_mass = 1.0
    is_contrastive = False

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
    bg_rollouts_test.extend(fg_rollouts)
    all_rollouts_test = bg_rollouts_test.copy()
    bg_success_queue = [0] * 3
    fg_success_queue = [0] * 3
    bg_converged = False

    # Setup agent
    nfq_net = ContrastiveNFQNetwork(
        state_dim=train_env_bg.state_dim, is_contrastive=is_contrastive
    )
    optimizer = optim.Adam(nfq_net.parameters(), lr=1e-1)
    nfq_agent = NFQAgent(nfq_net, optimizer)
    for ep in range(epoch + 1):

        if bg_converged:
            state_action_b, target_q_values, groups = nfq_agent.generate_pattern_set(
                fg_rollouts
            )
        else:
            state_action_b, target_q_values, groups = nfq_agent.generate_pattern_set(
                bg_rollouts
            )

        loss = nfq_agent.train((state_action_b, target_q_values, groups))

        eval_episode_length_fg = 0
        eval_episode_length_bg = 0
        eval_success_fg = False
        eval_success_bg = False
        if bg_converged:
            (
                eval_episode_length_fg,
                eval_success_fg,
                eval_episode_cost_fg,
            ) = nfq_agent.evaluate(eval_env_fg, render=False)
        else:
            (
                eval_episode_length_bg,
                eval_success_bg,
                eval_episode_cost_bg,
            ) = nfq_agent.evaluate(eval_env_bg, render=False)

        bg_success_queue = bg_success_queue[1:]
        bg_success_queue.append(1 if eval_success_bg else 0)

        fg_success_queue = fg_success_queue[1:]
        fg_success_queue.append(1 if eval_success_fg else 0)

        if sum(bg_success_queue) == 3:
            bg_converged = True

        if sum(fg_success_queue) == 3:
            if verbose:
                print("FG Converged")
            break

        if verbose:
            print(
                "Epoch: "
                + str(ep)
                + " BG Converged: "
                + str(bg_converged)
                + " Eval BG: "
                + str(eval_episode_length_bg)
                + " Eval FG: "
                + str(eval_episode_length_fg)
            )
    eval_env_bg.step_number = 0
    eval_env_fg.step_number = 0

    eval_env_bg.max_steps = 1000
    eval_env_fg.max_steps = 1000

    performance_fg = []
    performance_bg = []
    for it in range(evaluations):
        (
            eval_episode_length_bg,
            eval_success_bg,
            eval_episode_cost_bg,
        ) = nfq_agent.evaluate(eval_env_bg, False)
        performance_bg.append(eval_episode_length_bg)
        train_env_bg.close()
        eval_env_bg.close()

        (
            eval_episode_length_fg,
            eval_success_fg,
            eval_episode_cost_fg,
        ) = nfq_agent.evaluate(eval_env_fg, False)
        performance_fg.append(eval_episode_length_fg)
        train_env_fg.close()
        eval_env_fg.close()

    return nfq_agent


def transfer_learning(
    verbose=True,
    epoch=3000,
    init_experience=200,
    evaluations=5,
    force_left=5,
    random_seed=None,
):
    bg_cart_mass = 1.0
    fg_cart_mass = 1.0
    is_contrastive = False

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
    bg_rollouts_test.extend(fg_rollouts)
    all_rollouts_test = bg_rollouts_test.copy()

    bg_success_queue = [0] * 3
    fg_success_queue = [0] * 3
    bg_converged = False

    # Setup agent
    nfq_net = ContrastiveNFQNetwork(
        state_dim=train_env_bg.state_dim, is_contrastive=is_contrastive
    )

    optimizer = optim.Adam(nfq_net.parameters(), lr=1e-1)
    nfq_agent = NFQAgent(nfq_net, optimizer)
    nfq_agent._nfq_net.freeze_last_layers()
    for ep in range(epoch + 1):

        if bg_converged:
            state_action_b, target_q_values, groups = nfq_agent.generate_pattern_set(
                fg_rollouts
            )
        else:
            state_action_b, target_q_values, groups = nfq_agent.generate_pattern_set(
                bg_rollouts
            )

        loss = nfq_agent.train((state_action_b, target_q_values, groups))

        eval_episode_length_fg = 0
        eval_episode_length_bg = 0
        eval_success_fg = False
        eval_success_bg = False
        if bg_converged:
            (
                eval_episode_length_fg,
                eval_success_fg,
                eval_episode_cost_fg,
            ) = nfq_agent.evaluate(eval_env_fg, render=False)
        else:
            (
                eval_episode_length_bg,
                eval_success_bg,
                eval_episode_cost_bg,
            ) = nfq_agent.evaluate(eval_env_bg, render=False)

        bg_success_queue = bg_success_queue[1:]
        bg_success_queue.append(1 if eval_success_bg else 0)

        fg_success_queue = fg_success_queue[1:]
        fg_success_queue.append(1 if eval_success_fg else 0)

        if sum(bg_success_queue) == 3 or ep == 500:
            nfq_agent._nfq_net.unfreeze_last_layers()
            bg_converged = True

        if sum(fg_success_queue) == 3:
            if verbose:
                print("FG Converged")
            break
        if verbose:
            print(
                "Epoch: "
                + str(ep)
                + " BG Converged: "
                + str(bg_converged)
                + " Eval BG: "
                + str(eval_episode_length_bg)
                + " Eval FG: "
                + str(eval_episode_length_fg)
            )

    eval_env_bg.step_number = 0
    eval_env_fg.step_number = 0

    eval_env_bg.max_steps = 1000
    eval_env_fg.max_steps = 1000

    performance_fg = []
    performance_bg = []
    for it in range(evaluations):
        (
            eval_episode_length_bg,
            eval_success_bg,
            eval_episode_cost_bg,
        ) = nfq_agent.evaluate(eval_env_bg, True)
        performance_bg.append(eval_episode_length_bg)
        train_env_bg.close()
        eval_env_bg.close()

        (
            eval_episode_length_fg,
            eval_success_fg,
            eval_episode_cost_fg,
        ) = nfq_agent.evaluate(eval_env_fg, True)
        performance_fg.append(eval_episode_length_fg)
        train_env_fg.close()
        eval_env_fg.close()

    return nfq_agent