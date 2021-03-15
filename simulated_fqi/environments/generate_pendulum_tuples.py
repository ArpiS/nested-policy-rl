import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from os.path import join as pjoin
from pendulum import PendulumEnv



# fg and bg have different mass in this case

def tuples():
    ### Background
    pend = PendulumEnv(m=1.0)
    bg_tuples = pend.generate_tuples(n_iter=100, group="background")

    ## Foreground
    pend = PendulumEnv(m=5.0)
    fg_tuples = pend.generate_tuples(n_iter=100, group="foreground")
    return bg_tuples, fg_tuples

#import ipdb; ipdb.set_trace()





