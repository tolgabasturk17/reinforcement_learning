import gym
import numpy as np
import random
import matplotlib.pyplot as plt


env = gym.make("Taxi-v3", render_mode="human")

#Q-table
q_table = np.zeros([env.observation_space.n, env.action_space.n])
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
    state = state[0]
    reward_count = 0
    dropout_count = 0

    while True:

        # exploit vs explore to find action
        # %10 = explore, %90 exploit
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # action process and take reward / take observation
        next_state, reward, done, _, info = env.step(action)

        # Q Learning function
        old_value = q_table[state, action]
        next_max = np.argmax(q_table[next_state])
        next_value = (1-alpha) * old_value + alpha * (reward + gamma * next_max)

        # update Q-table
        q_table[state, action] = next_value

        #update state
        state = next_state

        # find wrong dropouts
        if reward == -10:
            dropout_count += 1

        if done:
            break;

        reward_count += reward

    if i % 10 == 0:
        dropout_list.append(dropout_count)
        reward_list.append(reward_count)
        print("Episode: {}, reward {}, wrong dropout {}".format(i, reward_count, dropout_count))

fig, axs = plt.subplots(1,2)
axs[0].plot(reward_list)
axs[0].set_xlabel("episode")
axs[0].set_ylabel("reward")

axs[0].plot(dropout_list)
axs[0].set_xlabel("episode")
axs[0].set_ylabel("dropout_count")

axs[0].grid(True)
axs[1].grid(True)

plt.show()