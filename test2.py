# Suppose these are some dummy transitions stored in a deque
from collections import deque
import random
# Each transition: (state, action, reward, next_state, done)
transitions = deque([
    ({"policy": [1, 2, 3, 4]}, 0, 1.0, {"policy": [2, 3, 4, 5]}, False),
    ({"policy": [2, 3, 4, 5]}, 1, 0.5, {"policy": [3, 4, 5, 6]}, True),
    ({"policy": [3, 4, 5, 6]}, 0, 0.2, {"policy": [4, 5, 6, 7]}, False),
])

# Sampling a batch (for simplicity, we'll just take all transitions)
batch = list(transitions)  # batch is a list of tuples

# Use *zip to separate each element of the tuples
# states, actions, rewards, next_states, dones = zip(*batch)
states, actions, rewards, next_states, dones = batch

buf = random.sample(batch, 2)

print(batch)
print(zip(*batch))
print(states)
