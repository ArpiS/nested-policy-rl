import configargparse
import torch
import torch.optim as optim

from environments import CartPoleRegulatorEnv
from environments import CartEnv
from environments import AcrobotEnv
from models.agents import NFQAgent
from models.networks import NFQNetwork, ContrastiveNFQNetwork
# from simulated_fqi import NFQAgent
# from simulated_fqi import NFQNetwork, ContrastiveNFQNetwork
from util import get_logger, close_logger, load_models, make_reproducible, save_models
import matplotlib.pyplot as plt
import numpy as np
import itertools


def main(verbose=True, is_contrastive=False):
    """Run NFQ."""
    # Setup hyperparameters
    parser = configargparse.ArgParser()
    parser.add("-c", "--config", required=True, is_config_file=True)
    parser.add("--EPOCH", type=int)
    parser.add("--TRAIN_ENV_MAX_STEPS", type=int)
    parser.add("--EVAL_ENV_MAX_STEPS", type=int)
    parser.add("--DISCOUNT", type=float)
    parser.add("--INIT_EXPERIENCE", type=int)
    parser.add("--INCREMENT_EXPERIENCE", action="store_true")
    parser.add("--HINT_TO_GOAL", action="store_true")
    parser.add("--RANDOM_SEED", type=int)
    parser.add("--TRAIN_RENDER", action="store_true")
    parser.add("--EVAL_RENDER", action="store_true")
    parser.add("--SAVE_PATH", type=str, default="")
    parser.add("--LOAD_PATH", type=str, default="")
    parser.add("--USE_TENSORBOARD", action="store_true")
    parser.add("--USE_WANDB", action="store_true")
    CONFIG = parser.parse_args()
    if not hasattr(CONFIG, "INCREMENT_EXPERIENCE"):
        CONFIG.INCREMENT_EXPERIENCE = False
    if not hasattr(CONFIG, "HINT_TO_GOAL"):
        CONFIG.HINT_TO_GOAL = False
    if not hasattr(CONFIG, "TRAIN_RENDER"):
        CONFIG.TRAIN_RENDER = False
    if not hasattr(CONFIG, "EVAL_RENDER"):
        CONFIG.EVAL_RENDER = False
    if not hasattr(CONFIG, "USE_TENSORBOARD"):
        CONFIG.USE_TENSORBOARD = False
    if not hasattr(CONFIG, "USE_WANDB"):
        CONFIG.USE_WANDB = False

    print()
    print("+--------------------------------+--------------------------------+")
    print("| Hyperparameters                | Value                          |")
    print("+--------------------------------+--------------------------------+")
    for arg in vars(CONFIG):
        print(
            "| {:30} | {:<30} |".format(
                arg, getattr(CONFIG, arg) if getattr(CONFIG, arg) is not None else ""
            )
        )
    print("+--------------------------------+--------------------------------+")
    print()

    # Log to File, Console, TensorBoard, W&B
    logger = get_logger()

    if CONFIG.USE_TENSORBOARD:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir="tensorboard_logs")
    if CONFIG.USE_WANDB:
        import wandb

        wandb.init(project="implementations-nfq", config=CONFIG)

    # Setup environment
    bg_cart_mass = 1.0
    fg_cart_mass = 1.0
    force_left = 0
    # train_env_bg = CartEnv(group=0, masscart=bg_cart_mass, mode="train", force_left=force_left)
    # train_env_fg = CartEnv(group=1, masscart=fg_cart_mass, mode="train", force_left=force_left)
    # eval_env_bg = CartEnv(group=0, masscart=bg_cart_mass, mode="eval", force_left=force_left)
    # eval_env_fg = CartEnv(group=1, masscart=fg_cart_mass, mode="eval", force_left=force_left)
    train_env_bg = CartPoleRegulatorEnv(group=0, masscart=bg_cart_mass, mode="train", force_left=force_left, is_contrastive=is_contrastive)
    train_env_fg = CartPoleRegulatorEnv(group=1, masscart=fg_cart_mass, mode="train", force_left=force_left, is_contrastive=is_contrastive)
    eval_env_bg = CartPoleRegulatorEnv(group=0, masscart=bg_cart_mass, mode="eval", force_left=force_left, is_contrastive=is_contrastive)
    eval_env_fg = CartPoleRegulatorEnv(group=1, masscart=fg_cart_mass, mode="eval", force_left=force_left, is_contrastive=is_contrastive)

    # Fix random seeds
    if CONFIG.RANDOM_SEED is not None:
        make_reproducible(CONFIG.RANDOM_SEED, use_numpy=True, use_torch=True)
        train_env_bg.seed(CONFIG.RANDOM_SEED)
        train_env_fg.seed(CONFIG.RANDOM_SEED)
        eval_env_bg.seed(CONFIG.RANDOM_SEED)
        eval_env_fg.seed(CONFIG.RANDOM_SEED)
    else:
        logger.warning("Running without a random seed: this run is NOT reproducible.")

    # Setup agent
    nfq_net = ContrastiveNFQNetwork(state_dim=train_env_bg.state_dim, is_contrastive=is_contrastive)
    # optimizer = optim.Rprop(nfq_net.parameters())

    if is_contrastive:
        optimizer = optim.Adam(itertools.chain(nfq_net.layers_shared.parameters(), nfq_net.layers_last_shared.parameters()), lr=1e-1)
    else:
        optimizer = optim.Adam(nfq_net.parameters(), lr=1e-1)
    
    nfq_agent = NFQAgent(nfq_net, optimizer)

    # Load trained agent
    if CONFIG.LOAD_PATH:
        load_models(CONFIG.LOAD_PATH, nfq_net=nfq_net, optimizer=optimizer)

    # NFQ Main loop
    # A set of transition samples denoted as D
    bg_rollouts = []
    fg_rollouts = []
    total_cost = 0
    if CONFIG.INIT_EXPERIENCE:
        for _ in range(CONFIG.INIT_EXPERIENCE):
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

    bg_success_queue = [0] * 3
    fg_success_queue = [0] * 3
    
    for epoch in range(CONFIG.EPOCH + 1):

        state_action_b, target_q_values, groups = nfq_agent.generate_pattern_set(all_rollouts)

        if not nfq_net.freeze_shared:
            loss = nfq_agent.train((state_action_b, target_q_values, groups))


        eval_episode_length_fg, eval_success_fg, eval_episode_cost_fg = 0, 0, 0

        if is_contrastive:
            if nfq_net.freeze_shared:
                eval_episode_length_fg, eval_success_fg, eval_episode_cost_fg = nfq_agent.evaluate(
                    eval_env_fg, render=False
                )
            else:
                eval_episode_length_bg, eval_success_bg, eval_episode_cost_bg = nfq_agent.evaluate(
                    eval_env_bg, render=False
                )
            
                
        else:
            eval_episode_length_bg, eval_success_bg, eval_episode_cost_bg = nfq_agent.evaluate(
                    eval_env_bg, render=False
                )
            eval_episode_length_fg, eval_success_fg, eval_episode_cost_fg = nfq_agent.evaluate(
                eval_env_fg, render=False
            )
        
        nfq_net.assert_correct_layers_frozen()

        bg_success_queue = bg_success_queue[1:]
        bg_success_queue.append(1 if eval_success_bg else 0)

        fg_success_queue = fg_success_queue[1:]
        fg_success_queue.append(1 if eval_success_fg else 0)

        if sum(bg_success_queue) == 3 and not nfq_net.freeze_shared:
            nfq_net.freeze_shared = True
            print("FREEZING SHARED")
            if is_contrastive:
                nfq_net.freeze_shared_layers()
                nfq_net.unfreeze_fg_layers()

                optimizer = optim.Adam(itertools.chain(nfq_net.layers_fg.parameters(), nfq_net.layers_last_fg.parameters()), lr=1e-1)
                nfq_agent._optimizer = optimizer

        # Print current status
        logger.info(
            "Epoch {:4d} | Eval BG {:4d} / {:4f} | Eval FG {:4d} / {:4f} | Train Loss {:.4f}".format(
                epoch, eval_episode_length_bg, eval_episode_cost_bg, eval_episode_length_fg, eval_episode_cost_fg, loss
            )
        )
        if CONFIG.USE_TENSORBOARD:
            writer.add_scalar("train/loss", loss, epoch)
            writer.add_scalar("eval/episode_length", eval_episode_length, epoch)
            writer.add_scalar("eval/episode_cost", eval_episode_cost, epoch)
        if CONFIG.USE_WANDB:
            wandb.log({"Train Loss": loss}, step=epoch)
            wandb.log(
                {"Evaluation Episode Length": eval_episode_length}, step=epoch
            )
            wandb.log({"Evaluation Episode Cost": eval_episode_cost}, step=epoch)

        if is_contrastive and sum(fg_success_queue) == 3:
            logger.info(
                "Epoch {:4d} | Total Cycles {:6d} | Total Cost {:4.2f}".format(
                    epoch, len(all_rollouts), total_cost
                )
            )
            if CONFIG.USE_TENSORBOARD:
                writer.add_scalar("summary/total_cycles", len(all_rollouts), epoch)
                writer.add_scalar("summary/total_cost", total_cost, epoch)
            if CONFIG.USE_WANDB:
                wandb.log({"Total Cycles": len(all_rollouts)}, step=epoch)
                wandb.log({"Total Cost": total_cost}, step=epoch)
            break

        if not is_contrastive and (sum(bg_success_queue) == 3 or sum(fg_success_queue) == 3):
            logger.info(
                "Epoch {:4d} | Total Cycles {:6d} | Total Cost {:4.2f}".format(
                    epoch, len(all_rollouts), total_cost
                )
            )
            if CONFIG.USE_TENSORBOARD:
                writer.add_scalar("summary/total_cycles", len(all_rollouts), epoch)
                writer.add_scalar("summary/total_cost", total_cost, epoch)
            if CONFIG.USE_WANDB:
                wandb.log({"Total Cycles": len(all_rollouts)}, step=epoch)
                wandb.log({"Total Cost": total_cost}, step=epoch)
            break

    # Save trained agent
    if CONFIG.SAVE_PATH:
        save_models(CONFIG.SAVE_PATH, nfq_net=nfq_net, optimizer=optimizer)

    eval_env_bg.step_number = 0
    eval_env_fg.step_number = 0
    
    eval_env_bg.max_steps = 1000
    eval_env_fg.max_steps = 1000

    # eval_env_bg.save_gif = True
    eval_episode_length_bg, eval_success_bg, eval_episode_cost_bg = nfq_agent.evaluate(eval_env_bg, render=False)
    # eval_env_bg.create_gif()

    print(eval_episode_length_bg, eval_success_bg)
    train_env_bg.close()
    eval_env_bg.close()

    # eval_env_fg.save_gif = True
    eval_episode_length_fg, eval_success_fg, eval_episode_cost_fg = nfq_agent.evaluate(eval_env_fg, render=False)
    # eval_env_fg.create_gif()

    print(eval_episode_length_fg, eval_success_fg)
    train_env_fg.close()
    eval_env_fg.close()
    #import ipdb; ipdb.set_trace()
    close_logger()
    return eval_episode_length_bg, eval_episode_length_fg


def run(verbose=True, is_contrastive=False, epoch=1000, train_env_max_steps=100, eval_env_max_steps=3000, discount=0.95, init_experience=200,
        increment_experience=0, hint_to_goal=0, evaluations=5, force_left=5, random_seed=1234):
    # Setup environment
    bg_cart_mass = 1.0
    fg_cart_mass = 1.0
    # train_env_bg = CartEnv(group=0, masscart=bg_cart_mass, mode="train", force_left=force_left)
    # train_env_fg = CartEnv(group=1, masscart=fg_cart_mass, mode="train", force_left=force_left)
    # eval_env_bg = CartEnv(group=0, masscart=bg_cart_mass, mode="eval", force_left=force_left)
    # eval_env_fg = CartEnv(group=1, masscart=fg_cart_mass, mode="eval", force_left=force_left)
    train_env_bg = CartPoleRegulatorEnv(group=0, masscart=bg_cart_mass, mode="train", force_left=force_left, is_contrastive=is_contrastive)
    train_env_fg = CartPoleRegulatorEnv(group=1, masscart=fg_cart_mass, mode="train", force_left=force_left, is_contrastive=is_contrastive)
    eval_env_bg = CartPoleRegulatorEnv(group=0, masscart=bg_cart_mass, mode="eval", force_left=force_left, is_contrastive=is_contrastive)
    eval_env_fg = CartPoleRegulatorEnv(group=1, masscart=fg_cart_mass, mode="eval", force_left=force_left, is_contrastive=is_contrastive)
    
    make_reproducible(random_seed, use_numpy=True, use_torch=True)
    train_env_bg.seed(random_seed)
    train_env_fg.seed(random_seed)
    eval_env_bg.seed(random_seed)
    eval_env_fg.seed(random_seed)

    # Log to File, Console, TensorBoard, W&B
    logger = get_logger()

    # Setup agent
    nfq_net = ContrastiveNFQNetwork(state_dim=train_env_bg.state_dim, is_contrastive=is_contrastive)
    # optimizer = optim.Rprop(nfq_net.parameters())

    if is_contrastive:
        optimizer = optim.Adam(itertools.chain(nfq_net.layers_shared.parameters(), nfq_net.layers_last_shared.parameters()), lr=1e-1)
    else:
        optimizer = optim.Adam(nfq_net.parameters(), lr=1e-1)

    nfq_agent = NFQAgent(nfq_net, optimizer)

    # NFQ Main loop
    # A set of transition samples denoted as D
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
    
    state_action_b, target_q_values, groups = nfq_agent.generate_pattern_set(all_rollouts_test)
    X_test = state_action_b
    test_groups = groups

    bg_success_queue = [0] * 3
    fg_success_queue = [0] * 3
    epochs_fg = 0
    eval_fg = 0
    for epoch in range(epoch + 1):

        state_action_b, target_q_values, groups = nfq_agent.generate_pattern_set(all_rollouts)
        X = state_action_b
        train_groups = groups
        
        if not nfq_net.freeze_shared:
            loss = nfq_agent.train((state_action_b, target_q_values, groups))
        
        eval_episode_length_fg, eval_success_fg, eval_episode_cost_fg = 0, 0, 0
        if nfq_net.freeze_shared:
            eval_fg += 1
            
            if eval_fg > 50:
                loss = nfq_agent.train((state_action_b, target_q_values, groups))

        if is_contrastive:
            # import ipdb; ipdb.set_trace()
            if nfq_net.freeze_shared:
                eval_episode_length_fg, eval_success_fg, eval_episode_cost_fg = nfq_agent.evaluate(
                    eval_env_fg, render=False
                )
                for param in nfq_net.layers_fg.parameters():
                    assert param.requires_grad == True
                for param in nfq_net.layers_last_fg.parameters():
                    assert param.requires_grad == True
                for param in nfq_net.layers_shared.parameters():
                    assert param.requires_grad == False
                for param in nfq_net.layers_last_shared.parameters():
                    assert param.requires_grad == False
            else:

                for param in nfq_net.layers_fg.parameters():
                    assert param.requires_grad == False
                for param in nfq_net.layers_last_fg.parameters():
                    assert param.requires_grad == False
                for param in nfq_net.layers_shared.parameters():
                    assert param.requires_grad == True
                for param in nfq_net.layers_last_shared.parameters():
                    assert param.requires_grad == True
                eval_episode_length_bg, eval_success_bg, eval_episode_cost_bg = nfq_agent.evaluate(
                    eval_env_bg, render=False
                )


        else:
            eval_episode_length_bg, eval_success_bg, eval_episode_cost_bg = nfq_agent.evaluate(
                eval_env_bg, render=False
            )
            eval_episode_length_fg, eval_success_fg, eval_episode_cost_fg = nfq_agent.evaluate(
                eval_env_fg, render=False
            )

        # bg_success_queue.pop()
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
            # else:
            #     for param in nfq_net.layers_fg.parameters():
            #         param.requires_grad = False
            #     for param in nfq_net.layers_last_fg.parameters():
            #         param.requires_grad = False

                optimizer = optim.Adam(itertools.chain(nfq_net.layers_fg.parameters(), nfq_net.layers_last_fg.parameters()), lr=1e-1)
                nfq_agent._optimizer = optimizer
            # break

        # Print current status
        if verbose:
            logger.info(
                # "Epoch {:4d} | Eval BG {:4d} / {:4f} | Eval FG {:4d} / {:4f} | Train Loss {:.4f}".format(
                #     epoch, eval_env_bg.success_step, eval_episode_cost_bg, eval_env_fg.success_step, eval_episode_cost_fg, loss
                # )
                "Epoch {:4d} | Eval BG {:4d} / {:4f} | Eval FG {:4d} / {:4f} | Train Loss {:.4f}".format(
                    epoch, eval_episode_length_bg, eval_episode_cost_bg, eval_episode_length_fg, eval_episode_cost_fg, loss
                )
            )
        if sum(fg_success_queue) == 3:
            printed_fg = True
            if verbose:
                logger.info(
                    "Epoch {:4d} | Total Cycles {:6d} | Total Cost {:4.2f}".format(
                        epoch, len(all_rollouts), total_cost
                    )
                )
            break

    eval_env_bg.step_number = 0
    eval_env_fg.step_number = 0

    eval_env_bg.max_steps = 1000
    eval_env_fg.max_steps = 1000

    performance = []
    num_steps_bg = []
    num_steps_fg = []
    total = 0
    for it in range(evaluations):

        # eval_env_bg.save_gif = True
        eval_episode_length_bg, eval_success_bg, eval_episode_cost_bg = nfq_agent.evaluate(eval_env_bg, True)
        # eval_env_bg.create_gif()
        if verbose:
            print(eval_episode_length_bg, eval_success_bg)
        num_steps_bg.append(eval_episode_length_bg)
        performance.append(eval_episode_length_bg)
        total += 1
        train_env_bg.close()
        eval_env_bg.close()

        # eval_env_fg.save_gif = True
        eval_episode_length_fg, eval_success_fg, eval_episode_cost_fg = nfq_agent.evaluate(eval_env_fg, True)
        # eval_env_fg.create_gif()
        if verbose:
            print(eval_episode_length_fg, eval_success_fg)
        num_steps_fg.append(eval_episode_length_fg)
        performance.append(eval_episode_length_fg)
        total += 1
        train_env_fg.close()
        eval_env_fg.close()
    print("Fg trained after " + str(epochs_fg) + " epochs")
    print("BG stayed up for steps: ", num_steps_bg)
    print("FG stayed up for steps: ", num_steps_fg)
    return performance
    
    
    

def warm_start(verbose=True, is_contrastive=False, epoch=1000, train_env_max_steps=100, eval_env_max_steps=3000, discount=0.95, init_experience=200,
        increment_experience=0, hint_to_goal=0, evaluations=5, force_left=5, random_seed=1234):
    # Setup environment
    bg_cart_mass = 1.0
    fg_cart_mass = 1.0
    init_experience = 400
    force_left = 5
    is_contrastive = False

    train_env_bg = CartPoleRegulatorEnv(group=0, masscart=bg_cart_mass, mode="train", force_left=force_left, is_contrastive=is_contrastive)
    train_env_fg = CartPoleRegulatorEnv(group=1, masscart=fg_cart_mass, mode="train", force_left=force_left, is_contrastive=is_contrastive)
    eval_env_bg = CartPoleRegulatorEnv(group=0, masscart=bg_cart_mass, mode="eval", force_left=force_left, is_contrastive=is_contrastive)
    eval_env_fg = CartPoleRegulatorEnv(group=1, masscart=fg_cart_mass, mode="eval", force_left=force_left, is_contrastive=is_contrastive)

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
    nfq_net = ContrastiveNFQNetwork(state_dim=train_env_bg.state_dim, is_contrastive=is_contrastive)
    optimizer = optim.Adam(nfq_net.parameters(), lr=1e-1)
    nfq_agent = NFQAgent(nfq_net, optimizer)
    for ep in range(epoch + 1):

        if bg_converged:
            state_action_b, target_q_values, groups = nfq_agent.generate_pattern_set(fg_rollouts)
        else:
            state_action_b, target_q_values, groups = nfq_agent.generate_pattern_set(bg_rollouts)

        loss = nfq_agent.train((state_action_b, target_q_values, groups))

        eval_episode_length_fg = 0
        eval_episode_length_bg = 0
        eval_success_fg = False
        eval_success_bg = False
        if bg_converged:
            eval_episode_length_fg, eval_success_fg, eval_episode_cost_fg = nfq_agent.evaluate(
                eval_env_fg, render=False
            )
        else:
            eval_episode_length_bg, eval_success_bg, eval_episode_cost_bg = nfq_agent.evaluate(
            eval_env_bg, render=False
            )

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
            print("Epoch: " + str(ep) + " BG Converged: " + str(bg_converged) + " Eval BG: " + str(eval_episode_length_bg)
                 + " Eval FG: " + str(eval_episode_length_fg))
    eval_env_bg.step_number = 0
    eval_env_fg.step_number = 0

    eval_env_bg.max_steps = 1000
    eval_env_fg.max_steps = 1000

    performance = []
    for it in range(evaluations):

        eval_episode_length_bg, eval_success_bg, eval_episode_cost_bg = nfq_agent.evaluate(eval_env_bg, True)
        performance.append(eval_episode_length_bg)
        train_env_bg.close()
        eval_env_bg.close()

        eval_episode_length_fg, eval_success_fg, eval_episode_cost_fg = nfq_agent.evaluate(eval_env_fg, True)
        performance.append(eval_episode_length_fg)
        train_env_fg.close()
        eval_env_fg.close()
   
    return performance
    
    


if __name__ == "__main__":
    # main(is_contrastive=True)

    n_repeats = 40
    fqi_lengths = []
    cfqi_lengths = []

    for n in range(n_repeats):
        eval_episode_length_bg, eval_episode_length_fg = main(is_contrastive=False)
        fqi_lengths.append(eval_episode_length_fg)

        eval_episode_length_bg, eval_episode_length_fg = main(is_contrastive=True)
        cfqi_lengths.append(eval_episode_length_fg)

        plt.boxplot([fqi_lengths, cfqi_lengths])
        plt.savefig("./intermediate_results.png")
        plt.close()
    # plt.show()

    import ipdb; ipdb.set_trace()



