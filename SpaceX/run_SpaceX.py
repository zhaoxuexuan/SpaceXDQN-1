"""
Policy Gradient, Reinforcement Learning.
The cart pole example
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import gym
import numpy as np
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

I_TEACH = 500
DISPLAY_REWARD_THRESHOLD = 2.5  # renders environment if total episode reward is greater than this threshold
RENDER = False  # rendering wastes time

env = gym.make('SpaceX-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.001,
    reward_decay=0.99,
    save_path=".\\network.nt",
     #output_graph=True,
)

for i_episode in range(2000):

    observation = env.reset() #observation=[x,x_dot]
    target = env.x_board, env.x_board_dot

    while True:
        # if i_episode >450 : RENDER = True
        if RENDER: env.render()

        if i_episode < I_TEACH:
            action = 1 if (observation[1]-target[1])*(observation[1]-target[1]) / 20.0 + observation[0] - target[0]< -8.75 else 0
        else:
            action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        RL.store_transition(observation, action, reward)

        if done:
            ep_rs_sum = sum(RL.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            #if running_reward > DISPLAY_REWARD_THRESHOLD and i_episode > I_TEACH * 2: RENDER = True     # rendering
            print("episode:", i_episode, "  reward:", running_reward, " real reward:", ep_rs_sum)

            vt = RL.learn()
            '''
            if i_episode == 0:
                plt.plot(vt)    # plot the episode vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            '''
            break
            
        observation = observation_

RL.save_net()

ep_rs_hist = []

for i_episode in range(100):

    observation = env.reset()
    reward_hist = []

    while True:
        #env.render()

        action = RL.choose_action(observation)

        observation, reward, done, info = env.step(action)
        reward_hist.append(reward)

        if done:
            reward_sum = np.sum(reward_hist)

            ep_rs_hist.append(reward_sum)
            print("episode:", i_episode, "  reward:", reward_sum)

            break

plt.plot(ep_rs_hist)    # plot the episode vt
plt.xlabel('episode num')
plt.ylabel('reward')
plt.show()
print("Average reward: ", np.average(ep_rs_hist))

input("hello")

ep_rs_hist = []

for i_episode in range(100):

    observation = env.reset()
    reward_hist = []

    while True:
        env.render()

        action = RL.test_action(observation)

        observation, reward, done, info = env.step(action)
        reward_hist.append(reward)

        if done:
            reward_sum = np.sum(reward_hist)

            ep_rs_hist.append(reward_sum)
            print("episode:", i_episode, "  reward:", reward_sum)

            break

plt.plot(ep_rs_hist)    # plot the episode vt
plt.xlabel('episode num')
plt.ylabel('reward')
plt.show()
print("Average reward: ", np.average(ep_rs_hist))