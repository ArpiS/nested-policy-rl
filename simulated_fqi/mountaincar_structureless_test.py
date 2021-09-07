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
perf_foreground = []
perf_background = []
for i in range(num_iter):
    print(str(i))
    perf_bg, perf_fg = fqi(epoch=1500, gravity=0.0025, verbose=True, is_contrastive=True, structureless=True)
    perf_foreground.append(perf_fg)
    perf_background.append(perf_bg)
sns.distplot(perf_foreground, label='Foreground Performance')
sns.distplot(perf_background, label='Background Performance')
plt.legend()
plt.xlabel("Average Reward Earned")
plt.title("Structureless Test: Gravity is the same in both groups")
# plt.show()
plt.savefig("./plots/mountaincar_structureless_test.png")
plt.close()


results = {
    "fg": perf_foreground,
    "bg": perf_background
}
with open("mountaincar_structureless_results.json", "w") as f:
    json.dump(results, f)

# import ipdb; ipdb.set_trace()
