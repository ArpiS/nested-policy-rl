import json
from train_mountaincar import fqi, warm_start, transfer_learning
import configargparse
import torch
import torch.optim as optim
import sys
sys.path.append('../')

from environments import MountainCarEnv, Continuous_MountainCarEnv
from models.agents import NFQAgent
from models.networks import NFQNetwork, ContrastiveNFQNetwork
from util import get_logger, close_logger, load_models, make_reproducible, save_models
import matplotlib.pyplot as plt
import numpy as np
import itertools
import seaborn as sns
import tqdm

num_iter = 1000
results = {}

GRAVITY = 0.004
N_EPOCHS = 1500

total_samples = 400
fg_sample_fractions = [0.1 * x for x in np.arange(1, 6)]

for i in fg_sample_fractions:
    results[i] = {}
    results[i]["fg_only"] = {}
    results[i]["cfqi"] = {}
    results[i]["fqi_joint"] = {}
    
for i in range(num_iter):

    for fg_sample_fraction in fg_sample_fractions:

        n_fg = int(total_samples * fg_sample_fraction)
        n_bg = int(total_samples - n_fg)
        
        # Only train/test on small set of foreground samples
        perf_bg, perf_fg = fqi(epoch=N_EPOCHS, verbose=False, is_contrastive=True, structureless=False, gravity=GRAVITY, fg_only=True, init_experience_bg=n_fg // 2,
            init_experience_fg=n_fg // 2)
        results[fg_sample_fraction]["fg_only"][i] = (perf_bg, perf_fg)

        # Use contrastive model with larger pool of background samples
        perf_bg, perf_fg = fqi(epoch=N_EPOCHS, is_contrastive=True,init_experience_bg=n_bg,init_experience_fg=n_fg,fg_only=False,verbose=False,gravity=GRAVITY)
        results[fg_sample_fraction]["cfqi"][i] = (perf_bg, perf_fg)

        # Use non-contrastive model with larger pool of background samples
        perf_bg, perf_fg = fqi(is_contrastive=False,init_experience_bg=n_bg,init_experience_fg=n_fg,fg_only=False,gravity=GRAVITY,epoch=N_EPOCHS,verbose=False,)
        results[fg_sample_fraction]["fqi_joint"][i] = (perf_bg, perf_fg)

        with open("mountaincar_class_imbalance_cfqi.json", "w") as f:
            json.dump(results, f)



            