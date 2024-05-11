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
        # acting
        pass

    def replay(self, batch_size):
        # training
        pass

    def adaptiveEGreedy(self):
        pass

if __name__ == "__main__":

    # initialize env and agent

    episodes = 100
    for e in range(episodes):

        # initialize environment

        while True:

            # act

            # step

            # remember

            # update state

            # replay

            #adjust epsilon

            if done:
                break


