"""
Classic SpaceX Landing system implemented by Luohao Wang.
"""

import logging
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)


class SpaceXEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.accle_mag = 0.5 * self.gravity
        self.board = [-15, 15]
        self.tau = 0.02  # seconds between state updates
        self.time = 0

        # Angle at which to fail the episode
        self.u_threshold = 0.5
        self.x_threshold = 50
        self.t_threshold = 10

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold,
            np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high)

        self._seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        state = self.state
        x, x_dot = state
        accle = self.accle_mag if action == 1 else -self.accle_mag
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * accle
        self.time += self.tau
        self.state = (x, x_dot)
        on_board = -self.board[0] < x < self.board[1]
        safe_speed = abs(x_dot) < self.u_threshold
        done = x < -self.x_threshold \
            or x > self.x_threshold \
            or self.time > self.t_threshold \
            or (on_board and safe_speed)
        done = bool(done)

        if not done:
            reward = -1.0
        elif on_board and safe_speed:
            reward = 1000
        else:
            reward = -100

        return np.array(self.state), reward, done, {}

    def _reset(self):
        # self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.state = np.array([-40, 0])
        self.steps_beyond_done = None
        self.time = 0
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        board_height = 10.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = self.board[0] * scale, self.board[1] * scale, board_height / 2, -board_height / 2
            board_area = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.boardtrans = rendering.Transform()
            board_area.add_attr(self.boardtrans)
            self.viewer.add_geom(board_area)
            masspt = rendering.make_circle(10)
            masspt.set_color(.5, .5, .8)
            self.masspttrans = rendering.Transform()
            masspt.add_attr(self.masspttrans)
            self.viewer.add_geom(masspt)

        if self.state is None:
            return None

        x = self.state
        boardx = screen_width / 2.0
        massptx = x[0] * scale + screen_width / 2.0
        self.boardtrans.set_translation(boardx, board_height)
        self.masspttrans.set_translation(massptx, (0.9 - self.time / self.t_threshold) * screen_height)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


if __name__ == '__main__':
    env = gym.make('SpaceX-v0')
    env.reset()
    while True:
        action = np.random.randint(0, 2)
        observation_, reward, done, info = env.step(action)
        env.render()
        if done:
            break
