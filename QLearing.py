import gym

env = gym.make("Taxi-v3", render_mode="human")

# Ortamı başlat
initial_state = env.reset()

# Ortamı render et
"""
blue = passenger
green = destination
yellow/read = empty taxi
"""

#%%
print("State space: ", env.observation_space)
print("Action space: ", env.action_space)

#state = env.encode(3,1,2,3)
#print("State number: ",state)

#env.s = state
#env.render()

#%%
"""
Actions:
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: drop off passenger
"""
#probability, next_state, reward, done
#print(env.P[331])

#%%

env.reset()

time_step = 0
total_reward = 0;
list_visualize = []
while True:

    #chose action
    action = env.action_space.sample()


    #perform action and get reward
    state, reward, done, _, info = env.step(action)

    #total reward
    total_reward += reward

    list_visualize.append({"frame" : env,
                            "state" : state,
                            "action" : action,
                            "reward" : reward,
                            "Total Reward" : total_reward})

    if done:
        break


import time

for i, frame in enumerate(list_visualize):
    print(frame["frame"])
    print("Timestep: ", i + 1)
    print("State: ", frame["state"])
    print("Action: ", frame["action"])
    print("Reward: ", frame["reward"])
    print("Total Reward: ", frame["Total Reward"])
    time.sleep(2)