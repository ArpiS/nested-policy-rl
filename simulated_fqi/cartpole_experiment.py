import configargparse
import torch
import torch.optim as optim

from environments import CartPoleRegulatorEnv
from environments import CartEnv
from environments import AcrobotEnv
from models.agents import NFQAgent
from models.networks import NFQNetwork, ContrastiveNFQNetwork
from util import get_logger, load_models, make_reproducible, save_models
import matplotlib.pyplot as plt
import numpy as np


def main():
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

    for _ in range(10):

        # Setup environment
        bg_cart_mass = 1.0
        fg_cart_mass = 1.0
        bg_pole_mass = 0.1
        fg_pole_mass = 0.1
        train_env_bg = CartPoleRegulatorEnv(mode="train", masscart=bg_cart_mass, masspole=bg_pole_mass, group=0)
        train_env_fg = CartPoleRegulatorEnv(mode="train", masscart=fg_cart_mass, masspole=fg_pole_mass, group=1)
        eval_env_bg = CartPoleRegulatorEnv(mode="eval", masscart=bg_cart_mass, masspole=bg_pole_mass, group=0)
        eval_env_fg = CartPoleRegulatorEnv(mode="eval", masscart=fg_cart_mass, masspole=fg_pole_mass, group=1)

        # bg_cart_mass = 1.0
        # fg_cart_mass = 1.0
        # train_env_bg = CartEnv(group=0, masscart=bg_cart_mass, mode="train")
        # train_env_fg = CartEnv(group=1, masscart=fg_cart_mass, mode="train")
        # eval_env_bg = CartEnv(group=0, masscart=bg_cart_mass, mode="eval")
        # eval_env_fg = CartEnv(group=1, masscart=fg_cart_mass, mode="eval")

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
        # nfq_net = NFQNetwork(state_dim=train_env_bg.state_dim)
        nfq_net = ContrastiveNFQNetwork(state_dim=train_env_bg.state_dim, is_contrastive=True)
        # optimizer = optim.Rprop(nfq_net.parameters())
        optimizer = optim.Adam(nfq_net.parameters(), lr=0.1)
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
        
        for epoch in range(CONFIG.EPOCH + 1):

            state_action_b, target_q_values, groups = nfq_agent.generate_pattern_set(all_rollouts)

            loss = nfq_agent.train((state_action_b, target_q_values, groups))

            eval_episode_length_bg, eval_success_bg, eval_episode_cost_bg = nfq_agent.evaluate(
                eval_env_bg, render=False
            )
            eval_episode_length_fg, eval_success_fg, eval_episode_cost_fg = nfq_agent.evaluate(
                eval_env_fg, render=False
            )

            # Print current status
            logger.info(
                # "Epoch {:4d} | Eval BG {:4d} / {:4f} | Eval FG {:4d} / {:4f} | Train Loss {:.4f}".format(
                #     epoch, eval_env_bg.success_step, eval_episode_cost_bg, eval_env_fg.success_step, eval_episode_cost_fg, loss
                # )
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

            if eval_success_bg and eval_success_fg:
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

        # angles = np.linspace(-2, 2, 200)
        # bg_actions = []
        # fg_actions = []
        # for x in angles:
        #     bg_a = nfq_agent.get_best_action([0, x, -0.1, 0], unique_actions=[0, 1], group=0)
        #     fg_a = nfq_agent.get_best_action([0, x, -0.1, 0], unique_actions=[0, 1], group=1)
        #     bg_actions.append(bg_a)
        #     fg_actions.append(fg_a)
        # print(np.all(np.array(fg_actions) == np.array(bg_actions)))
        # import ipdb; ipdb.set_trace()

        eval_env_bg.step_number = 0
        eval_env_fg.step_number = 0
        
        eval_episode_length_bg, eval_success_bg, eval_episode_cost_bg = nfq_agent.evaluate(eval_env_bg, False)
        print(eval_episode_length_bg, eval_success_bg)
        train_env_bg.close()
        eval_env_bg.close()
        eval_episode_length_fg, eval_success_fg, eval_episode_cost_fg = nfq_agent.evaluate(eval_env_fg, False)
        print(eval_episode_length_fg, eval_success_fg)
        train_env_fg.close()
        eval_env_fg.close()
        # import ipdb; ipdb.set_trace()
    
    
    
    


if __name__ == "__main__":
    main()
