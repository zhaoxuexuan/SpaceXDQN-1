"""
Deep Q network,
Using:
Tensorflow: 1.0
gym: 0.7.3
"""


import gym
from RL_brain import DeepQNetwork

env = gym.make('SpaceX-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  learning_rate=0.1, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001,)

total_steps = 0


for i_episode in range(1000):

    observation = env.reset()
    ep_r = 0
    while True:
        # env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        RL.store_transition(observation, action, reward, observation_)

        ep_r += reward
        if total_steps > 1000:
            RL.learn()

        if done:
            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 2),
                  ' epsilon: ', round(RL.epsilon, 2))
            break

        observation = observation_
        total_steps += 1

# Test Network
ep_ave = 0
for i_test in range(1):

    observation = env.reset()
    ep_r = 0
    while True:
        env.render()
        action = RL.test_action(observation)
        observation_, reward, done, info = env.step(action)
        ep_r += reward
        if done:
            ep_ave += ep_r
            print('Test round: ', i_test,
                  'ep_r: ', round(ep_r, 2))
            break

        observation = observation_

print('Ave reward: ', ep_ave / 1)
RL.plot_cost()