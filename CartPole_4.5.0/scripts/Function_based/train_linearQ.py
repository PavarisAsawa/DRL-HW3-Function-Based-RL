"""Script to train RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RL_Algorithm.Function_based.Linear_Q import *
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=3600, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=3600, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from datetime import datetime
import random
import json

import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
# from omni.isaac.lab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import CartPole.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

steps_done = 0

@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with stable-baselines agent."""
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # ==================================================================== #
    # ========================= Can be modified ========================== #

    # hyperparameters

    num_of_action: int = 7
    action_range: list = [-25, 25]
    learning_rate = 0.01
    initial_epsilon: float = 1.0
    epsilon_decay: float = 0.0003
    final_epsilon: float = 0.001
    discount = 0.95
    n_observations: int = 4
    n_episodes = 2500


    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    # if GPU is to be used
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    print("device: ", device)

    task_name = str(args_cli.task).split('-')[0]  # Stabilize, SwingUp
    Algorithm_name = "Linear_Q"
    experiment_name = "Linear_Q_3"
    fullpath = f"experiments/{Algorithm_name}/{experiment_name}"

    fullpath = f"experiments/{Algorithm_name}/{experiment_name}"
    writer = SummaryWriter(log_dir=f'runs/{Algorithm_name}/{experiment_name}')

    hyperparam = {
    "num_of_action": num_of_action,
    "action_range_min": action_range[0],
    "action_range_max": action_range[1],
    "learning_rate": learning_rate,
    "initial_epsilon": initial_epsilon,
    "epsilon_decay": epsilon_decay,
    "final_epsilon": final_epsilon,
    "discount": discount,
    "n_episodes": n_episodes,
    "describe" : "increase lr to 0.01 , increase epsilon decay"
    }
    #------------------------------------------------------------#
    # Dump Hyperparam
    os.makedirs(fullpath, exist_ok=True)
    # Save the JSON file
    with open(os.path.join(fullpath, "hyperparam.json"), "w") as f:
        json.dump(hyperparam, f, indent=4)
    #------------------------------------------------------------#

    agent = Linear_QN(
        num_of_action=num_of_action,
        action_range=action_range,
        learning_rate=learning_rate,
        initial_epsilon = initial_epsilon,
        epsilon_decay = epsilon_decay,
        final_epsilon = final_epsilon,
        discount_factor = discount,
    )

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        # with torch.inference_mode():
        
        for episode in tqdm(range(n_episodes)):
            cumulative_reward , step = agent.learn(env)

            writer.add_scalar("Reward/Episode", cumulative_reward, episode)
            writer.add_scalar("Time/Episode", step, episode)
            writer.add_scalar("Epsilon/Episode", agent.epsilon, episode)

        if episode % 100 == 0 or episode == n_episodes - 1:
            print(agent.epsilon)
        if (episode % 500 == 0) or (episode == n_episodes - 1):
                agent.save_w(fullpath, f"weight_{episode}")

        agent.save_w(fullpath, "weight")
        agent.save_reward(path=fullpath, filename="reward")
        agent.save_episode_duration(path=fullpath, filename="duration")

        print('Complete')
        agent.plot_durations(show_result=True)
        plt.ioff()
        plt.show()
            
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        break
    # ==================================================================== #

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()