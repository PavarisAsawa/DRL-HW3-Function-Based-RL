import numpy as np
from collections import defaultdict, namedtuple, deque
import random
from enum import Enum
import os
import json
import torch
import torch.nn as nn
import copy

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward' , 'done'))

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
    
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size = 1):
        """
        Initializes the replay buffer.

        Args:
            buffer_size (int): Maximum number of experiences the buffer can hold.
            batch_size (int): Number of experiences to sample per batch.
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, next_state, reward, done):
        """
        Adds an experience to the replay buffer.

        Args:
            state (Tensor): The current state of the environment.
            action (Tensor): The action taken at this state.
            reward (Tensor): The reward received after taking the action.
            next_state (Tensor): The next state resulting from the action.
            done (bool): Whether the episode has terminated.
        """
        self.memory.append(Transition(state, action, next_state, reward, done)) 

    def sample(self):
        """
        Samples a batch of experiences from the replay buffer.

        Returns:
            - state_batch: Batch of states.
            - action_batch: Batch of actions.
            - reward_batch: Batch of rewards.
            - next_state_batch: Batch of next states.
            - done_batch: Batch of terminal state flags.
        """
        a = random.sample(self.memory, self.batch_size)
        # print(a)
        return a

    def __len__(self):
        """
        Returns the current size of the replay buffer.

        Returns:
            int: The number of stored experiences.
        """
        return len(self.memory)


class BaseAlgorithm():
    """
    Base class for reinforcement learning algorithms.

    Attributes:
        num_of_action (int): Number of discrete actions available.
        action_range (list): Scale for continuous action mapping.
        discretize_state_scale (list): Scale factors for discretizing states.
        lr (float): Learning rate for updates.
        epsilon (float): Initial epsilon value for epsilon-greedy policy.
        epsilon_decay (float): Rate at which epsilon decays.
        final_epsilon (float): Minimum epsilon value allowed.
        discount_factor (float): Discount factor for future rewards.
        q_values (dict): Q-values for state-action pairs.
        n_values (dict): Count of state-action visits (for Monte Carlo method).
        training_error (list): Stores training errors for analysis.
    """

    def __init__(
        self,
        num_of_action: int = 2,
        action_range: list = [-2.0, 2.0],
        learning_rate: float = 1e-3,
        initial_epsilon: float = 1.0,
        epsilon_decay: float = 1e-3,
        final_epsilon: float = 0.001,
        discount_factor: float = 0.95,
        buffer_size: int = 1000,
        batch_size: int = 1,
    ):
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.num_of_action = num_of_action
        self.action_range = action_range  # [action_min, action_max]

        self.q_values = defaultdict(lambda: np.zeros(self.num_of_action))
        self.n_values = defaultdict(lambda: np.zeros(self.num_of_action))
        self.training_error = []
        self.rewards = []
        self.episode_durations = []

        self.w = np.zeros((4, num_of_action))
        self.memory = ReplayBuffer(buffer_size, batch_size)

    def q(self, obs, a=None):
        """Returns the linearly-estimated Q-value for a given state and action."""
        # Input obs : can be continuous value
        
        # ========= put your code here ========= #
        
        obs_val = obs['policy'][0].detach().cpu().numpy()
        if a==None:
            # Get q values from all action in state
            return np.dot(obs_val, self.w)
        else:
            # Get q values given action & state
            return np.dot(obs_val, self.w[:, a])
        # ====================================== #
        
    
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
        return torch.tensor([[action * ((self.action_range[1] - self.action_range[0]) / (self.num_of_action-1 )) + self.action_range[0]]])
        # ====================================== #
    
    def decay_epsilon(self):
        """
        Decay epsilon value to reduce exploration over time.
        """
        # ========= put your code here ========= #
        self.epsilon = max(self.final_epsilon, self.epsilon-self.epsilon_decay)
        # ====================================== #

    def save_w(self, path, filename):
        """
        Save weight parameters.
        """
        # ========= put your code here ========= #
        if not os.path.exists(path):
            os.makedirs(path)
        filepath = os.path.join(path, filename)
        np.save(filepath, self.w)
        # ====================================== #
            
    def load_w(self, path, filename):
        """
        Load weight parameters.
        """
        # ========= put your code here ========= #
        filepath = os.path.join(path, filename)
        if os.path.exists(filepath):
            self.w = np.load(filepath)
        # ====================================== #

    def save_loss(self, path, filename):
        """
        Save loss parameters.
        """
        # ========= put your code here ========= #
        if not os.path.exists(path):
            os.makedirs(path)
        filepath = os.path.join(path, filename)
        np.save(filepath, self.training_error)
        # ====================================== #
    
    def save_reward(self, path, filename):
        """
        Save cumulative_reward parameters.
        """
        # ========= put your code here ========= #
        if not os.path.exists(path):
            os.makedirs(path)
        filepath = os.path.join(path, filename)
        np.save(filepath, self.rewards)
        # ====================================== #
        
    def save_episode_duration(self, path, filename):
        """
        Save Episode Duration parameters.
        """
        # ========= put your code here ========= #
        if not os.path.exists(path):
            os.makedirs(path)
        filepath = os.path.join(path, filename)
        np.save(filepath, self.episode_durations)
        # ====================================== #

