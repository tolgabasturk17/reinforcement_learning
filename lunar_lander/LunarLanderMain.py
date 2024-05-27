import gym

if __name__ == '__main__':
    env = gym.make('LunarLander-v2', render_mode="human")

    n_games = 100

    for i in range(n_games):
        obs = env.reset()
        score = 0
        done = False
        while not done:
            env.render()
            action = env.action_space.sample()
            obs_, reward, done, truncated, info = env.step(action)
            score += reward
        print('episode ', i, 'score %.1f' % score)

    env.close()