
import gym
import numpy as np
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

env = gym.make('SpaceX-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.001,
    reward_decay=0.99,
    load_path=".\\network.nt",
    # output_graph=True,
)

ep_rs_hist = []

for i_episode in range(100):

    observation = env.reset()
    reward_hist = []

    while True:
        #env.render()

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