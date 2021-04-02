import configargparse
import torch
import torch.optim as optim

from environments import CartPoleRegulatorEnv
from models.agents import NFQAgent
from models.networks import NFQNetwork, ContrastiveNFQNetwork
from util import get_logger, load_models, make_reproducible, save_models
import matplotlib.pyplot as plt
import numpy as np

N_TRAINING_EPISODES_LIST = [50, 100, 200]
N_REPEATS = 10
N_TEST_EPISODES = 10
N_TRAINING_EPOCHS = 400

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

    logger = get_logger()

    # Setup environment
    bg_pole_length = 0.5
    fg_pole_length = 1.0
    successful_test_episodes_cnfqi = np.zeros((N_REPEATS, len(N_TRAINING_EPISODES_LIST)))
    successful_test_episodes_nfqi = np.zeros((N_REPEATS, len(N_TRAINING_EPISODES_LIST)))

    for is_contrastive in [False, True]:
        for repeat_idx in range(N_REPEATS):

            for train_idx, n_train in enumerate(N_TRAINING_EPISODES_LIST):
                train_env_bg = CartPoleRegulatorEnv(mode="train", length=bg_pole_length, group=0)
                train_env_fg = CartPoleRegulatorEnv(mode="train", length=fg_pole_length, group=1)
                eval_env_bg = CartPoleRegulatorEnv(mode="eval", length=bg_pole_length, group=0)
                eval_env_fg = CartPoleRegulatorEnv(mode="eval", length=fg_pole_length, group=1)

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
                nfq_net = ContrastiveNFQNetwork(state_dim=train_env_bg.state_dim, is_contrastive=is_contrastive)
                optimizer = optim.Rprop(nfq_net.parameters())
                nfq_agent = NFQAgent(nfq_net, optimizer)

                # NFQ Main loop
                # A set of transition samples denoted as D
                bg_rollouts = []
                fg_rollouts = []
                total_cost = 0
                if CONFIG.INIT_EXPERIENCE:
                    for _ in range(n_train):
                        rollout_bg, episode_cost = train_env_bg.generate_rollout(
                            None, render=False, group=0
                        )
                        rollout_fg, episode_cost = train_env_fg.generate_rollout(
                            None, render=False, group=1
                        )
                        bg_rollouts.extend(rollout_bg)
                        # fg_rollouts.extend(rollout_fg)
                        total_cost += episode_cost
                bg_rollouts.extend(fg_rollouts)
                all_rollouts = bg_rollouts.copy()
                
                for epoch in range(N_TRAINING_EPOCHS + 1):

                    state_action_b, target_q_values, groups = nfq_agent.generate_pattern_set(all_rollouts)

                    loss = nfq_agent.train((state_action_b, target_q_values, groups))

                    eval_episode_length_bg, eval_success_bg, eval_episode_cost_bg = nfq_agent.evaluate(
                        eval_env_bg, CONFIG.EVAL_RENDER
                    )
                    eval_episode_length_fg, eval_success_fg, eval_episode_cost_fg = nfq_agent.evaluate(
                        eval_env_fg, CONFIG.EVAL_RENDER
                    )

                    # Print current status
                    logger.info(
                            "Epoch {:4d} | BG: {:4d} | FG: {:4d} | Train Loss {:.4f}".format(
                                epoch, eval_episode_length_bg, eval_episode_length_fg, loss
                            )
                        )
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

                n_successful_episodes = 0
                for test_idx in range(N_TEST_EPISODES):
                    eval_episode_length_bg, eval_success_bg, eval_episode_cost_bg = nfq_agent.evaluate(eval_env_bg, False)
                    eval_episode_length_fg, eval_success_fg, eval_episode_cost_fg = nfq_agent.evaluate(eval_env_fg, False)
                    if eval_success_bg and eval_success_fg:
                        n_successful_episodes += 1

                if is_contrastive:
                    successful_test_episodes_cnfqi[repeat_idx, train_idx] = 1.0 * n_successful_episodes / N_TEST_EPISODES
                else:
                    successful_test_episodes_nfqi[repeat_idx, train_idx] = 1.0 * n_successful_episodes / N_TEST_EPISODES

                train_env_bg.close()
                eval_env_bg.close()
                train_env_fg.close()
                eval_env_fg.close()

    plt.figure(figsize=(7, 5))
    means = np.mean(successful_test_episodes_nfqi, axis=0)
    stddevs = np.std(successful_test_episodes_nfqi, axis=0)
    # plt.plot(N_TRAINING_EPISODES_LIST, means, c="black")
    # plt.fill_between(N_TRAINING_EPISODES_LIST, means - stddevs, means + stddevs, alpha=0.2)
    plt.errorbar(x=N_TRAINING_EPISODES_LIST, y=means, yerr=stddevs, label="NFQI")

    means = np.mean(successful_test_episodes_cnfqi, axis=0)
    stddevs = np.std(successful_test_episodes_cnfqi, axis=0)
    plt.errorbar(x=N_TRAINING_EPISODES_LIST, y=means, yerr=stddevs, label="CNFQI")
    plt.legend()
    plt.xlabel("Num. training episodes")
    plt.ylabel("Fraction successful test episodes")
    plt.tight_layout()
    plt.savefig("./cartpole_nfqi_vs_cnfqi.png")
    plt.show()
    import ipdb; ipdb.set_trace()
    
    
    
    


if __name__ == "__main__":
    main()
