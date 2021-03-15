import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from os.path import join as pjoin

ASSETS_DIR = "../../gym/gym/envs/classic_control/assets"

class PendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, g=10.0, m=1.):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.m = m
        self.l = 1.
        self.viewer = None

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), pjoin(ASSETS_DIR, "clockwise.png"))
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def generate_tuples(self, group, n_iter=101):

        if group not in ["background", "foreground"]:
            raise Exception("group must be a string: 'background' or 'foreground'")
        # group is a string: "background" or "foreground"

        STATE_DIM = 3
        states = np.zeros((n_iter, STATE_DIM))
        s_init = self.reset()
        states[0, :] = s_init
        costs = np.zeros(n_iter)
        actions = np.zeros(n_iter)

        for ii in range(1, n_iter):

            # Randomly sample an action
            a = self.action_space.sample()

            # Currently discretizes action to nearest integer
            a = np.rint(a)

            # Perform the action
            s, cost, _, _ = self.step(a)
            states[ii, :] = s
            actions[ii] = a
            costs[ii] = cost

        ## Form tuples
        tuples = []
        for ii in range(n_iter-1):

            s = states[ii, :]
            a = actions[ii]
            ns = states[ii+1, :]
            r = -costs[ii]

            # Tuples are (state, action, next state, reward, group, index)
            curr_tuple = (s, a, ns, np.asarray([r]), group, ii)
            tuples.append(curr_tuple)

        return tuples


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

if __name__ == "__main__":

    pend = PendulumEnv()

    STATE_DIM = 3
    n_iter = 1000
    states = np.zeros((n_iter, STATE_DIM))
    s_init = pend.reset()
    states[0, :] = s_init
    costs = np.zeros(n_iter)

    for ii in range(1, n_iter):

        # Randomly sample an action
        a = pend.action_space.sample()

        # Perform the action
        s, cost, _, _ = pend.step(a)
        states[ii, :] = s
        costs[ii] = cost

        # Render the current frame
        print(cost)
        pend.render()
        import ipdb; ipdb.set_trace()





