import gym
import numpy as np
from collections import deque
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random

class DQLAgent:

    def __init__(self, env):
        # parameters / hyper parameters
        pass

    def build_model(self):
        # neural network for deep q learning
        pass

    def remember(self, state, action, reward, next_state, done):
        # storage
        pass

    def act(self, state):
        # acting explore or explotation
        pass

    def replay(self, batch_size):
        # training
        pass

    def adaptiveEGreedy(self):
        pass

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
            next_state, reward, done, _ = env.step(action)
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


