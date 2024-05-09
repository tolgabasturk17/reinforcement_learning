import gym
import numpy as np
import random
import matplotlib.pyplot as plt


env = gym.make("Taxi-v3", render_mode="human")

#Q-table
q_table = np.zeros([env.observation_space.n , env.action_space.n])
#Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# Plotting Metrics
reward_list = []
dropout_list = []

episode_number = 10000
for i in range(1,episode_number):

    #initialize environment
    state = env.reset()

    reward_count = 0
    dropout_count = 0
    while True:

        # exploit vs explore to find action
        # %10 = explore, %90 exploit
        if random.uniform(0.1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # action process and take reward / take observation
        next_state, reward, done, _, info = env.step(action)

        # Q Learning function

        # update Q-table

        #update state

        # find wrong dropouts

        if done:
            break;

