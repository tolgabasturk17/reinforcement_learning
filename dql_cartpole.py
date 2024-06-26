import os
import gym
import numpy as np
from collections import deque
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random
import logging

logging.disable(logging.WARNING)

class DQLAgent:

    def __init__(self, env):
        # parameters / hyper parameters
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.gamma = 0.95
        self.learning_rate = 0.001
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.memory = deque(maxlen=1000)
        self.model = self.build_model()

    def build_model(self):
        # neural network for deep q learning
        model = keras.Sequential()
        model.add(Dense(48, input_dim = self.state_size, activation="tanh"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer = Adam(learning_rate = self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        # storage
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # acting explore or explotation
        if random.uniform(0,1) <= self.epsilon:
            return env.action_space.sample()
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])

    def replay(self, batch_size):
        # training
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            trained_target = self.model.predict(state)
            trained_target[0][action] = target
            self.model.fit(state, trained_target, verbose=0)

    def adaptiveEGreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay

#main method
if __name__ == "__main__":

    # initialize env and agent
    env = gym.make("CartPole-v0")
    agent = DQLAgent(env)

    batch_size = 16
    episodes = 100
    for e in range(episodes):

        # initialize environment
        state, _ = env.reset()

        state = np.reshape(state, (1,-1))
        time = 0
        while True:

            # act
            action = agent.act(state) #select an action

            # step
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1,4])

            # remember /storage
            agent.remember(state, action, reward, next_state, done)

            # update state
            state = next_state

            # replay
            agent.replay(batch_size)

            #adjust epsilon
            agent.adaptiveEGreedy()

            time += 1

            if done:
                print("Episode: {}, time: {}".format(e,time))
                break

# %% test
import time
trained_model = agent
state, _ = env.reset()
state = np.reshape(state, (1,-1))
time_t=0
while True:
    env.render()
    action = trained_model.act(state)
    next_state, reward, done, _, _ = env.step(action)
    next_state = np.reshape(next_state, [1, 4])
    state = next_state
    time_t +=1
    print(time_t)
    time.sleep(0.4)
    if done:
        break
print("Done")



