import random
import os
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

from torch.distributions.normal import Normal
from torch.nn.functional import mse_loss
from RL_Algorithm.RL_base_function import BaseAlgorithm
from .RolloutBuffer import *

class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=1e-4):
        """
        Actor network for policy approximation.

        Args:
            input_dim (int): Dimension of the state space.
            hidden_dim (int): Number of hidden units in layers.
            output_dim (int): Dimension of the action space.
            learning_rate (float, optional): Learning rate for optimization. Defaults to 1e-4.
        """
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim) # Input to hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) # hidden to hidden layer
        
        self.actor_head = nn.Linear(hidden_dim, output_dim) # hidden layer
        
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

        return actor_prob
class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, learning_rate=1e-4):
        """
        Actor network for policy approximation.

        Args:
            input_dim (int): Dimension of the state space.
            hidden_dim (int): Number of hidden units in layers.
            output_dim (int): Dimension of the action space.
            learning_rate (float, optional): Learning rate for optimization. Defaults to 1e-4.
        """
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim) # Input to hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) # hidden to hidden layer
        
        self.critic_head = nn.Linear(hidden_dim , 1)
        
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

        critic_out = self.critic_head(x)

        return critic_out

class PPO(BaseAlgorithm):
    rollout_buffer : RolloutBuffer
    actor : Actor
    critic : Critic
    def __init__(
            self,
            device=None,
            learning_rate : float = 0.01,
            discount_factor : float = 0.95,
            gae_lambda : float = 0.95,
            ent_coeff : float = 0.01,
            vf_coeff : float = 0.5,
            eps_clip : float = 0.2,
            vf_clip : float = 0.2,
            n_observations : int = 4,
            n_action : int = 7,
            action_range : list = [-5.0,5],
            actions_dim : int = 1,
            hidden_dim : int = 64,
            n_envs : int = 256,
            n_steps : int = 750,
            batch_size : int = 64,
            buffer_size : int = 4096,
            n_epochs : int = 1,
    ):
        self.device = device
        self.actor = Actor(n_observations, hidden_dim, actions_dim, learning_rate).to(device)
        self.critic = Critic(n_observations, hidden_dim, learning_rate).to(device)
        self.gae_lambda = gae_lambda
        self.rollout_buffer = RolloutBuffer(n_envs=n_envs , action_dim=actions_dim , buffer_size=buffer_size , gae_lambda=gae_lambda ,gamma=discount_factor , observation_dim=n_observations , device=device)
        self.eps_clip = eps_clip
        self.vf_clip = vf_clip
        self.n_envs = n_envs
        self.actions_dim = actions_dim
        self.ent_coeff = ent_coeff
        self.vf_coeff = vf_coeff
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.n_steps = n_steps

        self._last_obs = None
        self._last_episode_starts = None 

        self.optimizer = optim.AdamW(list(self.actor.parameters()) + list(self.critic.parameters()), lr=learning_rate, amsgrad=True)
        
        self.num_timesteps = 0

        super(PPO, self).__init__(
            num_of_action=n_action,
            action_range=action_range,
            discount_factor=discount_factor,
            learning_rate=learning_rate,
        )

        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display

    def select_action(self, prob_each_action) -> int:
        # Change to Probability Distribution
        dist = torch.distributions.Categorical(prob_each_action) # > Categorical(probs: torch.Size([1, 7]))
        action_idx = dist.sample() # > tensor([1], device='cuda:0')
        action_prob = dist.probs.gather(1, action_idx.unsqueeze(1)).squeeze(1)  # shape: [num_env]
        log_prob = dist.log_prob(action_idx) 
        entropy = dist.entropy()
        # [num_env] , [num_env , num_action] , [num_env , num_action] , [num_env] 
        return action_idx , action_prob ,log_prob , entropy 
    
    def scale_action(self, action):
        scale_factor = (self.action_range[1] - self.action_range[0]) / (self.num_of_action-1 )
        scaled_action = action * scale_factor + self.action_range[0]
        return scaled_action.view(-1, 1) 
    
    def evaluate_policy(self , action_idx , obs):
        # Actor Evaluate
        prob_each_action = self.actor(obs)
        dist = torch.distributions.Categorical(prob_each_action)
        log_prob = dist.log_prob(action_idx)
        entropy = dist.entropy()

        # Critic Evaluate 
        values = self.critic(obs)
        values = values.flatten()
        return values, log_prob, entropy
    
    def collect_rollout(self , env , roll_out_step):
        
        obs , _ = env.reset()
        training_flag = True
        episode_start = torch.zeros(self.n_envs, dtype=torch.bool)
        self.rollout_buffer.reset()

        steps_per_env = torch.zeros(self.n_envs, dtype=torch.int, device=obs['policy'].device)
        cumulative_reward_per_env = torch.zeros(self.n_envs, dtype=torch.float, device=self.device)
        time_step_buffer = deque(maxlen=self.n_envs)
        reward_buffer = deque(maxlen=self.n_envs)

        for stepping in range(roll_out_step):

            prob_each_action = self.actor(obs['policy'])
            action_idx , action_prob , log_prob , entropy  = self.select_action(prob_each_action=prob_each_action) # > tensor([4], device='cuda:0')
            values = self.critic(obs['policy'])
            # Execute action in the environment and observe next state and reward
            next_obs, reward, terminated, truncated, _ = env.step(self.scale_action(action_idx))  # Step Environmentscripts/Function_based/train.py --task Stabilize-Isaac-Cartpole-v0 
            dones = torch.logical_or(terminated, truncated)
            # print(obs['policy'].detach().cpu().numpy().shape)
            # print(action_idx.detach().cpu().numpy().shape)
            # print(reward)
            # print(episode_start)
            # print(values)
            # print(log_prob)
            # print(stepping)

            self.rollout_buffer.add(
                obs['policy'].detach().cpu().numpy(),  # type: ignore[arg-type]
                action_idx.detach().cpu().numpy(),
                reward.detach().cpu().numpy(),
                episode_start.cpu().numpy(),  # type: ignore[arg-type]
                values.detach().cpu().numpy(),
                log_prob.detach().cpu().numpy(),
            )

            obs = next_obs
            episode_start = dones
            cumulative_reward_per_env += reward

            active_envs = torch.logical_not(dones)
            steps_per_env[active_envs] += 1
            done_idx = torch.where(dones)[0]
            for index in done_idx:
                time_step_buffer.append(steps_per_env[index].item())
                reward_buffer.append(cumulative_reward_per_env[index].item())
                reward_avg = torch.mean(torch.tensor(reward_buffer, dtype=torch.float))
                time_avg = torch.mean(torch.tensor(time_step_buffer , dtype=torch.float))
            steps_per_env[done_idx] = 0
            cumulative_reward_per_env[done_idx] = 0

        with torch.no_grad():
            values = self.critic(next_obs['policy'])
        self.rollout_buffer.compute_returns_and_advantage(last_values=values , dones=dones)

        return training_flag , reward_avg.item() , time_avg.item()
    
    def learn(self , env , roll_out_step):
        # Collect Sample
        training_flag , reward , time = self.collect_rollout(env=env , roll_out_step=roll_out_step)
        
        # Use Sample
        entropy_losses = []
        pg_losses , values_losses = [] , []
        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                # print(actions.detach().cpu().numpy())
                values, log_prob, entropy = self.evaluate_policy(obs=rollout_data.observations , action_idx=actions)

                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean())/(advantages.std() + 1e-8) # Standardize


                # ratio between old and new policy,
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())

                value_pred = rollout_data.old_values + torch.clamp(
                    values - rollout_data.old_values, -self.vf_clip, self.vf_clip)
                
                value_loss = F.mse_loss(rollout_data.returns , value_pred)
                values_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coeff * entropy_loss + self.vf_coeff * value_loss 

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return loss.item() , value_loss.item() , policy_loss.item() , entropy_loss.item() , reward , time
    

    def save_net_weights(self, path, filename):
        """
        Save weight parameters.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        filepath = os.path.join(path, filename)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, filepath)
        
    def load_net_weights(self, path, filename):
        """
        Load weight parameters.
        """
        checkpoint = torch.load(os.path.join(path, filename))
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
