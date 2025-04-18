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

class RolloutBuffer():
    def __init__(self , buffer_size , n_envs):
        self.n_envs = n_envs
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)
        self.advantages = torch.tensor((self.buffer_size, self.n_envs), dtype=torch.float32)
    def add(self, state, action, reward, log_prob, values, done):
        # Detach to avoid carrying the computation graph
        self.memory.append((
            state.detach(), 
            action.detach(), 
            reward.detach(), 
            log_prob.detach(),
            values.detach(), 
            done.detach() if isinstance(done, torch.Tensor) else done
        ))
        
    def __len__(self):
        return len(self.memory)
    
    def sample_all_env(self , batch_size:int):
        '''
        Return random Transition with number of batch size from all environment 
        '''
        return random.sample(self.memory, batch_size)
    
    def sample_batch(self , batch_size:int):

        states, actions, rewards, log_probs_old, values, dones = zip(*self.memory)
        
        states        = torch.cat(states, dim=0) # > change to tensor
        actions       = torch.cat(actions, dim=0)
        rewards       = torch.cat(rewards, dim=0)
        log_probs_old = torch.cat(log_probs_old , dim=0)
        values        = torch.cat(values, dim=0)
        dones         = torch.cat(dones, dim=0)
        advantages    = self.advantages.flatten()

        random_indices = torch.randperm(len(states))[:batch_size]
        return states[random_indices] , actions[random_indices] , rewards[random_indices], log_probs_old[random_indices] , values[random_indices] , dones[random_indices] , advantages[random_indices]
    
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
    def __init__(self, 
                device = None, 
                num_of_action: int = 2,
                action_range: list = [-2.5, 2.5],
                n_observations: int = 4,
                hidden_dim = 256,
                dropout = 0.05, 
                learning_rate: float = 0.01,
                buffer_size: int = 256,
                batch_size: int = 1,
                discount_factor: float = 0.95,
                lamda : float = 1,
                nun_envs : int = 1,
                eps_clip : float = 0.2,
                critic_loss_coeff : float = 0.5,
                entropy_loss_coeff : float = 0.1,
                epoch : int = 20
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
        self.actor = Actor(n_observations, hidden_dim, num_of_action, learning_rate).to(device)
        self.critic = Critic(n_observations, hidden_dim, learning_rate).to(device)
        self.batch_size = batch_size
        self.lamda = lamda
        self.rollout_buffer = RolloutBuffer(buffer_size=buffer_size , n_envs =nun_envs)
        self.discount_factor = discount_factor
        self.eps_clip = eps_clip
        self.num_envs = nun_envs
        self.critic_loss_coeff = critic_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        self.epoch = epoch
        self.optimizer = optim.AdamW(list(self.actor.parameters()) + list(self.critic.parameters()), lr=learning_rate, amsgrad=True)

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
        # set up matplotlib
        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display

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
                - Log probability of the action taken.
                - Entropy of the action distribution.
        """
        # Change to Probability Distribution
        dist = torch.distributions.Categorical(prob_each_action) # > Categorical(probs: torch.Size([1, 7]))
        action_idx = dist.sample() # > tensor([1], device='cuda:0')
        action_prob = dist.probs.gather(1, action_idx.unsqueeze(1)).squeeze(1)  # shape: [num_env]
        log_prob = dist.log_prob(action_idx) 
        entropy = dist.entropy()
        # [num_env] , [num_env , num_action] , [num_env , num_action] , [num_env] 
        return action_idx , action_prob ,log_prob , entropy 

    def scale_action(self, action):
        """
        Maps a discrete action in range [0, n] to a continuous value in [action_min, action_max].

        Args:
            action (int): Discrete action in range [0, n].
            n (int): Number of discrete actions (inclusive range from 0 to n).
        
        Returns:
            torch.Tensor: Scaled action tensor.
        """
        # ========= put your code here ========= #
        # print("----------------")
        # print(action)
        scale_factor = (self.action_range[1] - self.action_range[0]) / (self.num_of_action-1 )
        scaled_action = action * scale_factor + self.action_range[0]
        return scaled_action.view(-1, 1) 
    
    def update_policy(self , memory):
        """
        Update the policy using the calculated loss.

        Returns:
            float: Loss value after the update.
        """
        states, actions, rewards, log_probs_old, values, dones , advantages = memory
        # for _ in range(self.epoch):
        values = self.critic(states).squeeze(-1) # > (batch size * num_env , 1)
        # advantage = self.calculate_advantage(rewards , dones , values.squeeze()) # > [] , [] , []
        
        values = (values-values.mean())/(values.std()+1e-8)

        returns = advantages + values

        probs = self.actor(states)                  # Get new action probabilities.
        dist = torch.distributions.Categorical(probs)
        log_probs_new = dist.log_prob(actions.squeeze())

        # Actor Loss
        ratio = torch.exp(log_probs_new - log_probs_old)
        surr1 = ratio*advantages
        surr2 = torch.clamp(ratio , 1.0-self.eps_clip , 1.0+self.eps_clip)*advantages
        actor_loss = -torch.min(surr1,surr2).mean()

        # Critic Loss
        critic_loss = F.mse_loss(values, returns)

        # Entropy bonus
        entrupy_bonus = dist.entropy().mean()

        # Final Loss
        loss = actor_loss + self.critic_loss_coeff*critic_loss + self.entropy_loss_coeff * entrupy_bonus
        # Perform backpropagation and optimizer step.
        self.optimizer.zero_grad()
        loss.backward()
        # Optionally clip gradients here if needed.
        self.optimizer.step()
        return loss.item() , actor_loss.item() , critic_loss.item() , entrupy_bonus.item()

    def calculate_advantage(self , rewards , dones , last_values):
        states, actions, rewards, log_probs_old, values , dones = zip(*self.rollout_buffer.memory)
        # Convert to tensors
        rewards = torch.stack(rewards).to(self.device)
        values = torch.stack(values).to(self.device).squeeze(2)
        dones = torch.stack(dones).to(self.device).float()
        last_values = last_values.flatten()
        
        # Dimension : [buffer_size , number_envs]
        # torch.Size([500, 64])
        # torch.Size([500, 64])
        # torch.Size([500, 64])
        # torch.Size([64])

        T_step, n_envs = rewards.shape
        advantages = torch.zeros((T_step, n_envs), dtype=torch.float32, device=self.device)
        gae = torch.zeros(n_envs, dtype=torch.float32, device=self.device)

        # print("debug.........")

        gae = torch.zeros(n_envs, dtype=torch.float32, device=self.device)

        for t in reversed(range(T_step)):
            mask = 1.0 - dones[t]  # [n_envs]
            next_value = last_values if t == T_step - 1 else values[t + 1]  # [n_envs]
            delta = rewards[t] + self.discount_factor * next_value * mask - values[t]
            gae = delta + self.discount_factor * self.lamda * mask * gae
            advantages[t] = gae
        advantages = (advantages - advantages.mean())/(advantages.std() + 1e-8)
        self.rollout_buffer.advantages = advantages

    
    # def compute_returns_and_advantage(self , last_values : torch.Tensor , dones : np.ndarray) -> None:
    #     last_values = last_values.clone().cpu().numpy().flatten()
    #     dones = dones.cpu().numpy()
    #     last_gae_lam = 0
    #     for step in reversed(range(self.buffer_size)):
    #         if step == self.buffer_size - 1: # Use real last value
    #             next_non_terminal = 1.0 - dones.astype(np.float32)
    #             next_values = last_values
    #         else:
    #             next_non_terminal = 1.0 - self.episode_starts[step + 1]
    #             next_values = self.values[step + 1]
    #         delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
    #         last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
    #         self.advantages[step] = last_gae_lam
    #         # print("debugging.....")

    def train(self , env , max_steps = 1000):
        obs , _  = env.reset()
        num_envs = obs['policy'].shape[0]

        steps_per_env = torch.zeros(num_envs, dtype=torch.int, device=obs['policy'].device)
        cumulative_reward_per_env = torch.zeros(num_envs, dtype=torch.float, device=self.device)

        time_step_buffer = deque(maxlen=10)
        reward_buffer = deque(maxlen=10)

        reward_avg = 0
        time_avg = 0

        loss = 0

        cumulative_reward = 0
        done = False
        # ====================================== #

        for step in range(max_steps):
            # Predict action from the policy network
            prob_each_action = self.actor(obs['policy']) 
            action_idx , action_prob , log_prob , entropy  = self.select_action(prob_each_action=prob_each_action) # > tensor([4], device='cuda:0')
            values = self.critic(obs['policy'])
            # Execute action in the environment and observe next state and reward
            next_obs, reward, terminated, truncated, _ = env.step(self.scale_action(action_idx))  # Step Environmentscripts/Function_based/train.py --task Stabilize-Isaac-Cartpole-v0 
            done = torch.logical_or(terminated, truncated)
            # Store the transition in memory
            self.rollout_buffer.add(state=obs['policy'],action=action_idx,reward=reward,log_prob=log_prob,values=values,done=done)
            

            # ====================================== #

            # Update state
            obs = next_obs
            active_envs = torch.logical_not(done)

            steps_per_env[active_envs] += 1
            done_idx = torch.where(done)[0]

            cumulative_reward_per_env += reward
            for index in done_idx:
                time_step_buffer.append(steps_per_env[index].item())
                reward_buffer.append(cumulative_reward_per_env[index].item())
                reward_avg = torch.mean(torch.tensor(reward_buffer, dtype=torch.float))
                time_avg = torch.mean(torch.tensor(time_step_buffer , dtype=torch.float))

            steps_per_env[done_idx] = 0
            cumulative_reward_per_env[done_idx] = 0

        last_val = self.critic(obs['policy'])
        advantage = self.calculate_advantage(reward, dones=done , last_values=last_val) # > [] , [] , []
        memory = self.rollout_buffer.sample_batch(self.batch_size)
        loss , actor_loss , critic_loss , entropy_bonus = self.update_policy(memory=memory)    
        # print("UPDATING POLICY!! ก'w'ก")

        # reward = 0
        # time_avg = 0
        # loss = 0
        return reward_avg , time_avg , loss , actor_loss , critic_loss , entropy_bonus

    def learn(self, env, max_steps=1000):
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
        reward_avg , timestep_avg , loss , actor_loss , critic_loss , entropy_bonus = self.train(env=env , max_steps=max_steps)
        self.training_error.append(loss)

        return reward_avg , timestep_avg , loss , actor_loss , critic_loss , entropy_bonus
        # self.plot_durations(timestep_avg)

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