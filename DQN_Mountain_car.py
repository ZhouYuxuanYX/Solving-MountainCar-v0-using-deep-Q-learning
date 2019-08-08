from collections import deque
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
import numpy as np
import gym
import matplotlib.pyplot as plt
import gc

#### Tips for speeding up the convergence
# Normalization: both state value and reward value
# Exploration: epsilon high enough in the early phase, step size high enough of each episode (in order to succeed at least once with random policy)

gc.enable()

class DQN:
    def __init__(self, env):
        self.env =env
        # Deque stands for a high performance container datatype: double-ended que
        # Memory is used for experience replay, bottle neck of the ram memory occupation
        self.memory = deque(maxlen=1000)
        # Define discount factor
        self.gamma = 0.99
        # Epsilon is used for exploration
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9  # epsilon**200 = 0.134
        self.learning_rate = 0.001

        self.model = self.create_model()
        # Target model that only update once a while to improve convergence
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Dense(48, input_dim=state_shape[0],
                activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error",
                optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, new_state, done):
        # Done stands for the last step of each episode, where no future value is available
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        inputs = np.zeros((batch_size, self.env.observation_space.shape[0]))  # 32, 80, 80, 4
        targets = np.zeros((batch_size, self.env.action_space.n))

        for i, sample in enumerate(samples):
                state, action, reward, new_state, done = sample
                inputs[i] = state
                target = self.target_model.predict(state)
                # No further action possible at the last step of a episode
                if done:
                    target[0][action] = reward
                else:
                # Target policy is greedy
                    Q_future = max(
                        self.target_model.predict(new_state)[0]
                    )
                    target[0][action] = reward + Q_future*self.gamma
                targets[i] = target
        # One-step update
        self.model.train_on_batch(inputs, targets)

    def target_train(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        # Sample from uniform distribution
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    def save_model(self, fn):
        self.model.save(fn)

def main():
    env = gym.make("MountainCar-v0")
    env._max_episode_steps = 500
    # env.render()  # This will cause the RAM usage to increase continuously!!!
    trials = 151
    trial_len = 500

    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    last_positions = []
    for trial in range(trials):
        cur_state = env.reset().reshape(1,2)
        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            new_state, reward, done, _ = env.step(action)
            # original shape of state is (2,), must be reshaped to be fed to keras model
            new_state = new_state.reshape(1,2)
            dqn_agent.remember(cur_state, action,
                               reward, new_state, done)

            dqn_agent.replay() # iterates prediction model
            if step % 50 == 0:
                dqn_agent.target_train() # iterates target model

            cur_state = new_state
            if done:
                break
        print("last position: {}".format(cur_state[0,0]))
        last_positions.append(cur_state[0,0])

        if step >= trial_len-1:
            print("Failed to complete at {} trial".format(trial))

        else:
            print("Completed in {} trials".format(trial))

        if trial % 50 == 0:
            # report every 50 trials, test 10 games to get average point score for statistics and verify if it is solved
            success = 0
            for i in range(10):
                obs = env.reset().reshape(1,2)
                done = False
                steps = 0
                while done != True:
                    action = np.argmax(dqn_agent.target_model.predict(obs)[0])
                    obs, rew, done, info = env.step(action)  # take step using selected action
                    obs = obs.reshape(1,2)
                    steps += 1
                if steps < trial_len-10:
                    success += 1
                print("step: {}".format(steps))
            print(success)
            print('Episode {} success rate: {}'.format(trial, success/10))

            if success > 8:
                print("Moutain car solved")
                dqn_agent.save_model("success.model")
                break

    plt.figure()
    plt.plot(last_positions)



if __name__ == "__main__":
    main()