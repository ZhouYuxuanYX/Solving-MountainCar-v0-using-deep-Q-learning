import gym
import numpy as np

#### Load the enviroment
env = gym.make('Frozenlake-v0')

#### Implement Q-Table learning algorithm

# initialize the table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])
# Set learning parameters
lr = .8
y = .95
num_episodes = 2000
# Create lists to contain total rewards and steps per episode
rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    # The Q-Table learning algorithm
    while j < 99:
        j += 1
        # Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        # Get new state and reward from environment
        sl, r, d, _ = env.step(a)
        # Update Q-table with new knowledge
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[sl,:]) - Q[s,a])