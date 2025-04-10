import random
import os
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

from torch.distributions.normal import Normal
from torch.nn.functional import mse_loss
from RL_Algorithm.RL_base_function import BaseAlgorithm

class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=1e-4):
        """
        Actor network for policy approximation.

        Args:
            input_dim (int): Dimension of the state space.
            hidden_dim (int): Number of hidden units in layers.
            output_dim (int): Dimension of the action space.
            learning_rate (float, optional): Learning rate for optimization. Defaults to 1e-4.
        """
        super(ActorCritic, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim) # Input to hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) # hidden to hidden layer
        
        self.actor_head = nn.Linear(hidden_dim, output_dim) # hidden layer
        self.critic_head = nn.Linear(hidden_dim , 1)
        
        self.softmax = nn.Softmax(dim=1)

        self.init_weights()

    def init_weights(self):
        """
        Initialize network weights using Xavier initialization for better convergence.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Xavier initialization
                nn.init.zeros_(m.bias)  # Initialize bias to 0
    def forward(self, state):
        """
        Forward pass for action selection.

        Args:
            state (Tensor): Current state of the environment.

        Returns:
            Tensor: Selected action values.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        actor_out = F.relu(self.actor_head(x))
        actor_prob = self.softmax(actor_out)
        critic_out = self.critic_head(x)

        return actor_prob , critic_out

class PPO(BaseAlgorithm):
    def __init__(self, 
                device = None, 
                num_of_action: int = 2,
                action_range: list = [-2.5, 2.5],
                n_observations: int = 4,
                hidden_dim = 256,
                dropout = 0.05, 
                learning_rate: float = 0.01,
                tau: float = 0.005,
                buffer_size: int = 256,
                batch_size: int = 1,
                discount_factor: float = 0.95,
                ):
        """
        Actor-Critic algorithm implementation.

        Args:
            device (str): Device to run the model on ('cpu' or 'cuda').
            num_of_action (int, optional): Number of possible actions. Defaults to 2.
            action_range (list, optional): Range of action values. Defaults to [-2.5, 2.5].
            n_observations (int, optional): Number of observations in state. Defaults to 4.
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 256.
            learning_rate (float, optional): Learning rate. Defaults to 0.01.
            tau (float, optional): Soft update parameter. Defaults to 0.005.
            discount_factor (float, optional): Discount factor for Q-learning. Defaults to 0.95.
            batch_size (int, optional): Size of training batches. Defaults to 1.
            buffer_size (int, optional): Replay buffer size. Defaults to 256.
        """
        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.device = device
        self.actor_critic = ActorCritic(n_observations, hidden_dim, num_of_action, learning_rate).to(device)
        self.batch_size = batch_size
        self.tau = tau
        self.update_target_networks(tau=1)  # initialize target networks

        # Experiment with different values and configurations to see how they affect the training process.
        # Remember to document any changes you make and analyze their impact on the agent's performance.

        pass
        # ====================================== #

        super(PPO, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )

    def select_action(self, prob_each_action, noise=0.0) -> int:
        """
        Selects an action based on the current policy with optional exploration noise.
        
        Args:
        state (Tensor): The current state of the environment. [[n1,n2,n3,n4,..nn]]
        noise (float, optional): The standard deviation of noise for exploration. Defaults to 0.0.

        Returns:
            Tuple[Tensor, Tensor]: 
                - scaled_action: The final action after scaling.
            Tensor:
                - Probabiblity from action : dim : tensor([n])
        """
        # Change to Probability Distribution
        prob_cat = torch.distributions.Categorical(prob_each_action) # > Categorical(probs: torch.Size([1, 7]))
        action_idx = prob_cat.sample() # > tensor([1], device='cuda:0')
        return action_idx , prob_cat.probs[0][action_idx]
    
    def calculate_stepwise_returns(self, rewards):
        """
        Compute stepwise returns for the trajectory.

        Args:
            rewards (list): List of rewards obtained in the episode.
        
        Returns:
            Tensor: Normalized stepwise returns. # Dim = [1]
        """
        stepwise_return = 0
        stepwise_return_arr = []
        for r in reversed(rewards):
            stepwise_return = stepwise_return*self.discount_factor + r
            stepwise_return_arr.append(stepwise_return)
        tensor_norm = F.normalize(input=torch.tensor(list(reversed(stepwise_return_arr))),dim=0)
        return tensor_norm # > tensor([-0.1740, -0.1021, 0.3525,  0.4109,  0.4675,  0.5201])
    
    def calculate_advantage(self , rewards , values):
        stepwise_return = self.calculate_stepwise_returns(rewards = rewards)
        pass

    def generate_sample(self, batch_size):
        """
        Generates a batch sample from memory for training.

        Returns:
            Tuple: A tuple containing:
                - state_batch (Tensor): The batch of current states.
                - action_batch (Tensor): The batch of actions taken.
                - reward_batch (Tensor): The batch of rewards received.
                - next_state_batch (Tensor): The batch of next states received.
                - done_batch (Tensor): The batch of dones received.
        """
        # Ensure there are enough samples in memory before proceeding
        # ========= put your code here ========= #
        # Sample a batch from memory
        batch = self.memory.sample()
        # ====================================== #
        
        # Sample a batch from memory
        # ========= put your code here ========= #
        pass
        # ====================================== #

    def calculate_loss(self, states, actions, rewards, next_states, dones):
        """
        Computes the loss for policy optimization.

        Args:
            - states (Tensor): The batch of current states.
            - actions (Tensor): The batch of actions taken.
            - rewards (Tensor): The batch of rewards received.
            - next_states (Tensor): The batch of next states received.
            - dones (Tensor): The batch of dones received.

        Returns:
            Tensor: Computed critic & actor loss.
        """
        # ========= put your code here ========= #
        # Update Critic

        # Gradient clipping for critic

        # Update Actor

        # Gradient clipping for actor

        pass
        # ====================================== #

    def update_policy(self):
        """
        Update the policy using the calculated loss.

        Returns:
            float: Loss value after the update.
        """
        # ========= put your code here ========= #
        sample = self.generate_sample(self.batch_size)
        if sample is None:
            return
        states, actions, rewards, next_states, dones = sample

        # Normalize rewards (optional but often helpful)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Compute critic and actor loss
        critic_loss, actor_loss = self.calculate_loss(states, actions, rewards, next_states, dones)
        
        # Backpropagate and update critic network parameters

        # Backpropagate and update actor network parameters
        # ====================================== #


    def update_target_networks(self, tau=None):
        """
        Perform soft update of target networks using Polyak averaging.

        Args:
            tau (float, optional): Update rate. Defaults to self.tau.
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #
    def generate_trajectory(self , env):
        obs , _  = env.reset()
        state_hist = []
        reward_hist = []
        action_hist = []
        timestep = 0
        cumulative_reward = 0
        done = False
        # ====================================== #

        while not done:
            # Predict action from the policy network
            prob_each_action , v_value = self.actor_critic(obs['policy'])
            action_idx , action_prob = self.select_action(prob_each_action=prob_each_action) # > tensor([4], device='cuda:0')
            # Execute action in the environment and observe next state and reward
            next_obs, reward, terminated, truncated, _ = env.step(self.scale_action(action_idx.item()))  # Step Environment
            reward_value = reward.item() # > int : 1
            terminated_value = terminated.item() 
            cumulative_reward += reward_value
            done = terminated or truncated

            # Store the transition in memory
            self.memory.add(state=obs,action=action_idx,next_state=next_obs,reward=reward_value,done=done)

            # ====================================== #

            # Update state
            obs = next_obs
            timestep += 1
            if done:
                self.plot_durations(timestep)
                break
        self.calculate_advantage()
        return 0

    def learn(self, env, max_steps=0, num_agents=1, noise_scale=0.1, noise_decay=0.99):
        """
        Train the agent on a single step.

        Args:
            env: The environment in which the agent interacts.
            max_steps (int): Maximum number of steps per episode.
            num_agents (int): Number of agents in the environment.
            noise_scale (float, optional): Initial exploration noise level. Defaults to 0.1.
            noise_decay (float, optional): Factor by which noise decreases per step. Defaults to 0.99.
        """

        # ===== Initialize trajectory collection variables ===== #
        # Reset environment to get initial state (tensor)
        # Track total episode return (float)
        # Flag to indicate episode termination (boolean)
        # Step counter (int)
        # ========= put your code here ========= #
        self.generate_trajectory(env=env)

    def save_net_weights(self, path, filename):
        """
        Save weight parameters.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        filepath = os.path.join(path, filename)
        torch.save(self.actor_critic.state_dict(), filepath)
        
    def load_net_weights(self, path, filename):
        """
        Load weight parameters.
        """
        self.actor_critic.load_state_dict(torch.load(os.path.join(path, filename)))

    # ================================================================================== #
    def plot_durations(self, timestep=None, show_result=False):
        if timestep is not None:
            self.episode_durations.append(timestep)

        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if self.is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
    # ================================================================================== #