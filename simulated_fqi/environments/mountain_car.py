"""
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""
import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
import torch

ASSETS_DIR = "../../gym/gym/envs/classic_control/assets"


class MountainCarEnv(gym.Env):
    """
    Description:
        The agent (a car) is started at the bottom of a valley. For any given
        state the agent may choose to accelerate to the left, right or cease
        any acceleration.
    Source:
        The environment appeared first in Andrew Moore's PhD Thesis (1990).
    Observation:
        Type: Box(2)
        Num    Observation               Min            Max
        0      Car Position              -1.2           0.6
        1      Car Velocity              -0.07          0.07
    Actions:
        Type: Discrete(3)
        Num    Action
        #0      Accelerate to the Left
        #1      Don't accelerate
        #2      Accelerate to the Right
        0      Accelerate to the Left
        #1      Don't accelerate
        1      Accelerate to the Right
        Note: This does not affect the amount of velocity affected by the
        gravitational pull acting on the car.
    Reward:
         Reward of 100 is awarded if the agent reached the flag (position = 0.5)
         on top of the mountain.
         Reward of 0 is awarded if the position of the agent is less than 0.5.
    Starting State:
         The position of the car is assigned a uniform random value in
         [-0.75 , 0.5]. (Used to be -0.6 to -0.4)
         The starting velocity of the car is always assigned to 0.
    Episode Termination:
         The car position is more than 0.5
         Episode length is greater than 200
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, goal_velocity=0, group=0):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = goal_velocity
        self.state_dim = 2
        
        self.force = 0.001
        self.gravity = 0.0025
        self.group = group

        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)

        self.viewer = None
        # Render the car
        # Run some of experiments for cartpole
        if self.group == 1:
            self.unique_actions = np.array([-4, 5])
        elif self.group == 0:
            self.unique_actions = np.array([-2, 3])
        #self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #assert self.action_space.contains(action), "%r (%s) invalid" % (
        #    action,
        #    type(action),
        #)

        position, velocity = self.state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0

        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        if position >= self.goal_position:
            reward = 100
        else:
            reward = 0

        self.state = (position, velocity)
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.75, high=0.5), 0.0])
        return np.array(self.state)
    def reset_cheat(self):
        self.state = np.array([self.np_random.uniform(low=-0.75, high=0.5), 0.0])
        return np.array(self.state)

    def _height(self, xs):
        return np.sin(3 * xs) * 0.45 + 0.55
    
    def get_goal_pattern_set(self, size: int = 100, group=0):
        """Use hint-to-goal heuristic to clamp network output.
        Parameters
        ----------
        size : int
            The size of the goal pattern set to generate.
        Returns
        -------
        pattern_set : tuple of np.ndarray
            Pattern set to train the NFQ network.
        """
        goal_state_action_b = [
            np.array(
                [
                    np.random.uniform(0.5, 0.55),
                    np.random.uniform(self.goal_velocity, self.goal_velocity + 0.1),
                    np.random.choice([4, -4])
                ]
            )
            for _ in range(size)
        ]
        goal_target_q_values = np.zeros(size)
        groups = np.asarray([group]*size)
        group_b = torch.FloatTensor(groups).unsqueeze(1)
        
        return goal_state_action_b, goal_target_q_values, group_b

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(0.5, 0.5, 0.5)
            frontwheel.add_attr(
                rendering.Transform(translation=(carwidth / 4, clearance))
            )
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(
                rendering.Transform(translation=(-carwidth / 4, clearance))
            )
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position - self.min_position) * scale
            flagy1 = self._height(self.goal_position) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(0.8, 0.8, 0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation(
            (pos - self.min_position) * scale, self._height(pos) * scale
        )
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def get_keys_to_action(self):
        # Control with left and right arrow keys.
        return {(): 1, (276,): 0, (275,): 2, (275, 276): 1}

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    
    def generate_rollout(
        self,
        agent= None,
        render= False,
        rollout_length= 50,
        group = 1, dataset='train'):
        """
        Generate rollout using given action selection function.
        If a network is not given, generate random rollout instead.
        Parameters
        ----------
        get_best_action : Callable
            Greedy policy.
        render: bool
            If true, render environment.
        Returns
        -------
        rollout : List of Tuple
            Generated rollout.
        episode_cost : float
            Cumulative cost throughout the episode.
        """
        rollout = []
        episode_cost = 0

        if dataset == 'train':
            obs = self.reset_cheat()
        else:
            obs = self.reset()

        info = {"time_limit": False}
        for ii in range(rollout_length):
            if agent is not None:
                action = agent.get_best_action(obs, self.unique_actions, group)
            else:
                action = np.random.choice(self.unique_actions)
            next_obs, cost, done, info = self.step(action)
            rollout.append(
                (obs.squeeze(), action, cost, next_obs.squeeze(), done, group)
            )
            episode_cost += cost
            obs = next_obs
            if done:
                break
            # import ipdb; ipdb.set_trace()

            if render:
                self.render()

        return rollout, episode_cost


if __name__ == "__main__":

    car = MountainCarEnv()
    car.reset()
    n_iter = 2000

    for ii in range(1, n_iter):

        # Randomly sample an action
        a = car.action_space.sample()

        # Perform the action
        s, reward, _, _ = car.step(a)

        # Render the current frame
        print(reward)
        car.render()
        # import ipdb; ipdb.set_trace()
