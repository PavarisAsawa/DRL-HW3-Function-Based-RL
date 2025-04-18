from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base_function import BaseAlgorithm


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
from collections import namedtuple, deque
import random
import matplotlib
import matplotlib.pyplot as plt
import os

class MC_REINFORCE_network(nn.Module):
    """
    Neural network for the MC_REINFORCE algorithm.
    
    Args:
        n_observations (int): Number of input features.
        hidden_size (int): Number of hidden neurons.
        n_actions (int): Number of possible actions.
        dropout (float): Dropout rate for regularization.
    """

    def __init__(self, n_observations, hidden_size, n_actions, dropout):
        super(MC_REINFORCE_network, self).__init__()
        # ========= put your code here ========= #
        self.fc1 = nn.Linear(n_observations, hidden_size) # Input layer
        self.fc2 = nn.Linear(hidden_size, n_actions) # hidden layer
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)
        # ====================================== #

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Output tensor representing action probabilities.
        """
        # ========= put your code here ========= #
        # input layer to hidden
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # hidden layer to softmax
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # softmax to output
        x = self.softmax(x)
        return x
        # ====================================== #

class MC_REINFORCE(BaseAlgorithm):
    def __init__(
            self,
            device = None,
            num_of_action: int = 2,
            action_range: list = [-2.5, 2.5],
            n_observations: int = 4,
            hidden_dim: int = 64,
            dropout: float = 0.5,
            learning_rate: float = 0.01,
            discount_factor: float = 0.95,
    ) -> None:
        """
        Initialize the CartPole Agent.

        Args:
            learning_rate (float): The learning rate for updating Q-values.
            initial_epsilon (float): The initial exploration rate.
            epsilon_decay (float): The rate at which epsilon decays over time.
            final_epsilon (float): The final exploration rate.
            discount_factor (float, optional): The discount factor for future rewards. Defaults to 0.95.
        """     

        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.LR = learning_rate

        self.policy_net = MC_REINFORCE_network(n_observations, hidden_dim, num_of_action, dropout).to(device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate)

        self.device = device
        self.steps_done = 0

        self.episode_durations = []

        # Experiment with different values and configurations to see how they affect the training process.
        # Remember to document any changes you make and analyze their impact on the agent's performance.

        pass
        # ====================================== #

        super(MC_REINFORCE, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )

        # set up matplotlib
        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display

        plt.ion()
    
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
        return tensor_norm.tolist() # > tensor([-0.1740, -0.1021, 0.3525,  0.4109,  0.4675,  0.5201])


    def generate_trajectory(self, env):
        """
        Generate a trajectory by interacting with the environment.

        Args:
            env: The environment object.
        
        Returns:
            Tuple: (timestep ,episode_return, stepwise_returns, log_prob_actions, trajectory)
        """
        # ===== Initialize trajectory collection variables ===== #
        # Reset environment to get initial state (tensor)
        # Store state-action-reward history (list)
        # Store log probabilities of actions (list)
        # Store rewards at each step (list)
        # Track total episode return (float)
        # Flag to indicate episode termination (boolean)
        # Step counter (int)
        # ========= put your code here ========= #
        obs , _  = env.reset()
        state_hist = []
        reward_hist = []
        action_hist = []
        log_prob_action_hist = []
        episode_return_hist = 0
        timestep = 0
        cumulative_reward = 0
        done = False
        # ====================================== #
        
        # ===== Collect trajectory through agent-environment interaction ===== #
        # In Episode
        while not done:
            
            # Predict action from the policy network
            # State into policy to return probability of each action
            prob_each_action = self.policy_net(obs['policy']) # > tensor([[0.1380, 0.1534, 0.1328, 0.1328, 0.1656, 0.1328, 0.1446]],device='cuda:0', grad_fn=<SoftmaxBackward0>)
            # Change to Probability Distribution
            prob_cat = torch.distributions.Categorical(prob_each_action) # > Categorical(probs: torch.Size([1, 7]))
            action_idx = prob_cat.sample() # > tensor([1], device='cuda:0')

            # Execute action in the environment and observe next state and reward
            next_obs, reward, terminated, truncated, _ = env.step(self.scale_action(action_idx))  # Step Environment
            reward_value = reward.item() # > int : 1
            terminated_value = terminated.item() 
            cumulative_reward += reward_value
            done = terminated or truncated

            # Store action log probability reward and trajectory history
            reward_hist.append(reward_value)
            state_hist.append(obs)
            log_prob_action_hist.append(prob_cat.log_prob(action_idx)) # Collect in list and reduce dimension and change to list
            
            # Update state
            obs = next_obs
            timestep += 1
            if done:
                self.plot_durations(timestep)
                break

        # ===== Stack log_prob_actions &  stepwise_returns ===== #
        stepwise_returns = self.calculate_stepwise_returns(rewards=reward_hist)
        loss = self.calculate_loss(stepwise_returns=stepwise_returns , log_prob_actions=log_prob_action_hist).item()
        self.training_error.append(loss)
        self.episode_durations.append(timestep)
        self.rewards.append(cumulative_reward)
        return (cumulative_reward , stepwise_returns , log_prob_action_hist , state_hist)
    
    def calculate_loss(self, stepwise_returns, log_prob_actions):
        """
        Compute the loss for policy optimization.
        Args:
            stepwise_returns (List): Stepwise returns for the trajectory. : Dim list = [n]
            log_prob_actions (tensor): Log probabilities of actions taken. : Dim list = [n] : n is tensor contain with prob
        
        Returns:
            Tensor: Computed loss.
        """
        loss = 0
        # สมมุติว่า log_prob_actions กับ stepwise_returns มีความสัมพันธ์ 1-to-1
        for t in range(len(log_prob_actions)):
            loss += -(log_prob_actions[t] * stepwise_returns[t])
            # loss += -sum(log_prob_actions[t]*stepwise_returns[t])
        loss = loss/len(stepwise_returns)
        return loss # > tensor(2.5966) : Scalar
    
    def update_policy(self, stepwise_returns, log_prob_actions):
        """
        Update the policy using the calculated loss.

        Args:
            stepwise_returns (Tensor): Stepwise returns.
            log_prob_actions (Tensor): Log probabilities of actions taken.
        
        Returns:
            float: Loss value after the update.
        """
        loss = self.calculate_loss(stepwise_returns=stepwise_returns , log_prob_actions=log_prob_actions).unsqueeze(0) # get tensor loss value
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def learn(self, env):
        """
        Train the agent on a single episode.

        Args:
            env: The environment to train in.
        
        Returns:
            Tuple: (episode_return, loss, trajectory)
        """
        # ========= put your code here ========= #
        self.policy_net.train()
        episode_return, stepwise_returns, log_prob_actions, trajectory = self.generate_trajectory(env)
        loss = self.update_policy(stepwise_returns, log_prob_actions)
        return episode_return, loss, trajectory
        # ====================================== #

    def save_net_weights(self, path, filename):
        """
        Save weight parameters.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        filepath = os.path.join(path, filename)
        torch.save(self.policy_net.state_dict(), filepath)
        
    def load_net_weights(self, path, filename):
        """
        Load weight parameters.
        """
        self.policy_net.load_state_dict(torch.load(os.path.join(path, filename)))

    # Consider modifying this function to visualize other aspects of the training process.
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
