from DQN_Mountain_car import DQN
import gym
from keras.models import load_model
import numpy as np
import time

def act(state,model):
    return np.argmax(model.predict(state)[0])

env = gym.make("MountainCar-v0")
trials = 10
trial_len = 500
model = load_model("success.model")

for trial in range(trials):
    cur_state = env.reset().reshape(1, 2)
    env.render()
    time.sleep(3)
    for step in range(trial_len):
        action = act(cur_state, model)
        new_state, reward, done, _ = env.step(action)

        print(reward)
        new_state = new_state.reshape(1,2)
        cur_state = new_state

        if done:
            break
    if step >= 199:
        print("Failed to complete trial")

    else:
        print("Completed in {} trials".format(trial))