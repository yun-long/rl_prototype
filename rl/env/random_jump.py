import numpy as np
from gym import spaces
import gym
from gym.utils import seeding
import math

class RandomJumpEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.60
        self.goal_position = 0.4
        #
        self.low_state = self.min_position
        self.high_state = self.max_position
        #
        self.action_space = spaces.Box(self.min_action, self.max_action, shape= (1,))
        self.observation_space = spaces.Box(self.min_position, self.max_position, shape=(1,))
        #
        self.viewer = None
        #
        self._seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = [self.np_random.uniform(low=-.90, high=-.80)]
        # self.state = [-1.2]
        return np.array(self.state)

    def _step(self, action):
        noise = np.random.uniform(low=-0.1,high=0.1)
        position = self.state[0]
        #
        force = min(max(action[0], self.min_action), self.max_action)
        # force = action
        position += (force + noise)
        #
        done = bool(position >= (self.goal_position-0.05) and position <= (self.goal_position+0.05))
        #
        # cost = np.square(self.goal_position - position)
        cost = np.square(self.goal_position - position) + math.pow(action[0],2)*0.1
        reward = -cost
        if done:
            reward = 1
        # reward = np.atleast_1d(reward)
        self.state = [min(max(position, self.min_position), self.max_position)]
        #
        return self.state[0], reward, done, {}

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        self._height = 0
        screen_width = 600
        screen_height = 400
        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carwidth=40
        carheight=20
        flagy1 = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = np.zeros(100)
            ys[:] = flagy1
            xys = list(zip((xs-self.min_position)*scale, ys))
            # self.track = rendering.Line((self.min_position*scale, flagy1), (self.max_position*scale, flagy1))
            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position)*scale
            # flagy1 = self._height(self.goal_position)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(.8,.8,0)
            self.viewer.add_geom(flag)

        # print(self.state)

        pos = self.state[0]
        self.cartrans.set_translation((pos-self.min_position)*scale, 20)
        # self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

