from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base_function import BaseAlgorithm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import random
import matplotlib
import matplotlib.pyplot as plt
import os

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward' , 'done'))

class DQN_network(nn.Module):
    """
    Neural network model for the Deep Q-Network algorithm.
    
    Args:
        n_observations (int): Number of input features.
        hidden_size (int): Number of hidden neurons.
        n_actions (int): Number of possible actions.
        dropout (float): Dropout rate for regularization.
    """
    def __init__(self, n_observations, hidden_size, n_actions, dropout):
        super(DQN_network, self).__init__()
        # ========= put your code here ========= #
        self.fc1 = nn.Linear(n_observations, hidden_size) # Input layer
        self.fc2 = nn.Linear(hidden_size, hidden_size) # hidden layer
        self.fc3 = nn.Linear(hidden_size, n_actions) # output layer
        self.dropout = nn.Dropout(dropout)
        
        # ====================================== #

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input state tensor.
        
        Returns:
            Tensor: Q-value estimates for each action.
        """
        # ========= put your code here ========= #
        val = x
        # input layer
        val = F.relu(self.fc1(val))
        val = self.dropout(val)
        
        # hidden layer
        val = F.relu(self.fc2(val))
        val = self.dropout(val)
        
        # output layer
        return self.fc3(val)
        # ====================================== #

class DQN(BaseAlgorithm):
    def __init__(
            self,
            device = None,
            num_of_action: int = 2,
            action_range: list = [-2.5, 2.5],
            n_observations: int = 4,
            hidden_dim: int = 64,
            dropout: float = 0.5,
            learning_rate: float = 0.01,
            tau: float = 0.005,
            initial_epsilon: float = 1.0,
            epsilon_decay: float = 1e-3,
            final_epsilon: float = 0.001,
            discount_factor: float = 0.95,
            buffer_size: int = 1000,
            batch_size: int = 1,
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
        self.policy_net = DQN_network(n_observations, hidden_dim, num_of_action, dropout).to(device)
        self.target_net = DQN_network(n_observations, hidden_dim, num_of_action, dropout).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.device = device
        self.steps_done = 0
        self.num_of_action = num_of_action
        self.tau = tau

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)

        self.buffer_size = buffer_size
        self.batch_size = batch_size

        # Experiment with different values and configurations to see how they affect the training process.
        # Remember to document any changes you make and analyze their impact on the agent's performance.

        # ====================================== #

        super(DQN, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,  
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )

        # set up matplotlib
        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display

        plt.ion()

    def select_action(self, state):
        """
        Select an action based on an epsilon-greedy policy.
        
        Args:
            state (Tensor): The current state of the environment.
        
        Returns:
            Tensor: The selected action.
        """
        # ========= put your code here ========= #
        if np.random.rand() > self.epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices
        else:
            return torch.tensor([random.randint(0, self.num_of_action-1)] , device=self.device)
                
            
        # ====================================== #

    def calculate_loss(self, non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch):
        """
        Computes the loss for policy optimization.

        Args:
            non_final_mask (Tensor): Mask indicating which states are non-final.
            non_final_next_states (Tensor): The next states that are not terminal.
            state_batch (Tensor): Batch of current states.
            action_batch (Tensor): Batch of actions taken.
            reward_batch (Tensor): Batch of received rewards.
        
        Returns:
            Tensor: Computed loss.
        """
        # ========= put your code here ========= #
        
        # Compute Q(s_t, a) using Policy Network
        q = self.policy_net(state_batch).gather(1, action_batch) # [batch_size, 1]

        # Compute V(s_t+1) for all next states using Target Network
        q_next = torch.zeros(size=(self.batch_size , self.num_of_action), device=self.device)
        if non_final_next_states.size(0) > 0:
            q_next_values = self.target_net(non_final_next_states).detach()
            q_next[non_final_mask.squeeze()] = q_next_values # Define Next Q value from next state , squeeze make dimension [batch_size , 1] to [batch_size]
        q_expected = (torch.max(q_next , dim=1)[0].unsqueeze(1) * self.discount_factor) + reward_batch # Find Maximum Q Value over action : Dimension

        loss = F.mse_loss(q_expected,q)
        return loss

        # ====================================== #

    def generate_sample(self, batch_size):
        """
        Generates a batch sample from memory for training.

        Returns:
            Tuple: A tuple containing:
                - non_final_mask (Tensor): A boolean mask indicating which states are non-final.
                - non_final_next_states (Tensor): The next states that are not terminal.
                - state_batch (Tensor): The batch of current states.
                - action_batch (Tensor): The batch of actions taken.
                - reward_batch (Tensor): The batch of rewards received.
        """
        # Ensure there are enough samples in memory before proceeding
        # sample for training with batch size
        if len(self.memory) < batch_size:
            return
        batch = self.memory.sample()         
        # ========= put your code here ========= #)
        state_batch = torch.stack([torch.tensor(batch[i].state, dtype=torch.float) for i in range(self.batch_size)]).to(self.device)
        next_states_batch = torch.stack([torch.tensor(batch[i].next_state, dtype=torch.float) for i in range(self.batch_size)]).to(self.device)
        action_batch = torch.stack([torch.tensor(batch[i].action, dtype=torch.int64) for i in range(self.batch_size)]).to(self.device)
        reward_batch = torch.stack([torch.tensor(batch[i].reward, dtype=torch.float) for i in range(self.batch_size)]).to(self.device)
        non_final_mask = torch.stack([torch.tensor(not batch[i].done, dtype=torch.bool) for i in range(self.batch_size)]).to(self.device)
        non_final_next_states = next_states_batch[non_final_mask]
        # Return All dimension : [batch_size , 1]
        return (non_final_mask.unsqueeze(1), non_final_next_states.squeeze(1), state_batch.squeeze(1), action_batch, reward_batch.unsqueeze(1))
        # ====================================== #

    def update_policy(self):
        """
        Update the policy using the calculated loss.

        Returns:
            float: Loss value after the update.
        """
        # Generate a sample batch
        sample = self.generate_sample(self.batch_size)
        if sample is None:
            return
        non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch = sample
        # Compute loss
        loss = self.calculate_loss(non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch)

        # Perform gradient descent step
        # ========= put your code here ========= #
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
        # ====================================== #

    def update_target_networks(self):
        """
        Soft update of target network weights using Polyak averaging.
        """
        # Retrieve the state dictionaries (weights) of both networks
        # ========= put your code here ========= #
        target_net_state_dict = self.target_net.state_dict() # get target network weights
        policy_net_state_dict = self.policy_net.state_dict()
        # ====================================== #
        
        # Apply the soft update rule to each parameter in the target network
        # ========= put your code here ========= #
        # for key in target_net_state_dict:
        for key in target_net_state_dict:
            target_net_state_dict[key] = self.tau * policy_net_state_dict[key] + (1.0 - self.tau) * target_net_state_dict[key]
        # ====================================== #
        
        # Load the updated weights into the target network
        # ========= put your code here ========= #
        self.target_net.load_state_dict(target_net_state_dict)
        # ====================================== #

    def learn(self, env):
        """
        Train the agent on a single step.

        Args:
            env: The environment to train in.
        """

        # ===== Initialize trajectory collection variables ===== #
        # Reset environment to get initial state (tensor)
        # Track total episode return (float)
        # Flag to indicate episode termination (boolean)
        # Step counter (int)
        # ========= put your code here ========= #
        obs , _  = env.reset()
        cumulative_reward = 0.0
        done = False
        step = 0
        # ====================================== #

        while not done:
            # Predict action from the policy network
            # ========= put your code here ========= #

            action = self.select_action(obs['policy'])
            # ====================================== #

            # Execute action in the environment and observe next state and reward
            # ========= put your code here ========= #
            next_obs, reward, terminated, truncated, _ = env.step(self.scale_action(action))

            reward_value = reward.item()
            terminated_value = terminated.item() 
            cumulative_reward += reward_value
            done = terminated or truncated
            # ====================================== #

            # Store the transition in memory
            # ========= put your code here ========= #
            self.memory.add(obs['policy'], action, next_obs['policy'], reward_value, done)
            # ====================================== #
            
            obs = next_obs
            
            # Update state
            # Perform one step of the optimization (on the policy network) and save training error
            loss = self.update_policy()
            self.training_error.append(loss)
            # Soft update of the target network's weights
            self.update_target_networks()
            
            

            # Decaying Epsilon
            step += 1

            if done:
                self.plot_durations(step)
                self.rewards.append(cumulative_reward)
                break
    
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
    
        # ====================================== #

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