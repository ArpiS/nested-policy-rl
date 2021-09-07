import json
from train_mountaincar import fqi, warm_start, transfer_learning
import configargparse
import torch
import torch.optim as optim
import scipy
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

initial_g = 0.0025

# num_iter = 1000
# results = {}
# for i in range(0, 5):
#     results[i] = {}
#     results[i]['cfqi'] = {}
#     results[i]['fqi'] = {}
#     results[i]['warm_start'] = {}
#     results[i]['transfer_learning'] = {}
    

# N_EPOCHS = 1500
# x = []
# for i in range(num_iter):
#     for f in range(0, 5):
#         gravity = initial_g + f*0.0005
#         print(str(gravity))
#         x.append(gravity)
        
#         perf_bg, perf_fg = fqi(epoch=N_EPOCHS, verbose=False, is_contrastive=True, structureless=True, gravity=gravity)
#         results[f]['cfqi'][i] = (perf_fg, perf_bg)
        
#         perf_bg, perf_fg = fqi(epoch=N_EPOCHS, verbose=False, is_contrastive=False, structureless=True, gravity=gravity)
#         results[f]['fqi'][i] = (perf_fg, perf_bg)
        
#         perf_bg, perf_fg = warm_start(epoch=N_EPOCHS, verbose=False, structureless=True, gravity=gravity)
#         results[f]['warm_start'][i] = (perf_fg, perf_bg)
        
#         perf_bg, perf_fg = transfer_learning(epoch=N_EPOCHS, verbose=False, structureless=True, gravity=gravity)
#         results[f]['transfer_learning'][i] = (perf_fg, perf_bg)
        
        
        
        
#     with open('gravity_v_performance.json', 'w') as f:
#         json.dump(results, f) 


with open("./gravity_v_performance.json") as f:
    results = json.load(f)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def plot_performance(results, ds='bg'):

    x = []
    c_success = []
    f_success = []
    w_success = []
    t_success = []
    c_errs = []
    f_errs = []
    w_errs = []
    t_errs = []
    if ds == 'bg':
        ind = 1
    else:
        ind = 0
    for i in np.arange(0, 5).astype("str"):

        f = i.astype("float")
        gravity = initial_g + f*0.0005
        x.append(gravity)

        cfqi_perf = []
        fqi_perf = []
        ws_perf = []
        tl_perf = []
        for key in results[i]['fqi']:
            fqi_perf.append(results[i]['fqi'][key][ind])
        for key in results[i]['cfqi']:
            cfqi_perf.append(results[i]['cfqi'][key][ind])
        for key in results[i]['warm_start']:
            ws_perf.append(results[i]['warm_start'][key][ind])
        for key in results[i]['transfer_learning']:
            tl_perf.append(results[i]['transfer_learning'][key][ind])

        c_success.append(np.mean(cfqi_perf))
        f_success.append(np.mean(fqi_perf))
        w_success.append(np.mean(ws_perf))
        t_success.append(np.mean(tl_perf))
        m, h = mean_confidence_interval(cfqi_perf)
        c_errs.append(h)
        m, h = mean_confidence_interval(fqi_perf)
        f_errs.append(h)
        m, h = mean_confidence_interval(ws_perf)
        w_errs.append(h)
        m, h = mean_confidence_interval(tl_perf)
        t_errs.append(h) 

    plt.figure(figsize=(10, 4))
    # import ipdb; ipdb.set_trace()
    sns.scatterplot(x, c_success, label='CFQI')

    plt.errorbar(x, c_success ,yerr=c_errs, linestyle="None")
    sns.scatterplot(x, f_success, label='FQI')
    plt.errorbar(x, f_success ,yerr=f_errs, linestyle="None")
    sns.scatterplot(x, w_success, label='Warm Start')
    plt.errorbar(x, w_success ,yerr=w_errs, linestyle="None")
    sns.scatterplot(x, t_success, label='Transfer Learning')
    plt.errorbar(x, t_success ,yerr=t_errs, linestyle="None")
    if ds == 'bg':
        plt.title("Background Dataset: Performance of CFQI, FQI, Warm Start, Transfer Learning when gravity is modified")
    else:
        plt.title("Foreground Dataset: Performance of CFQI, FQI, Warm Start, Transfer Learning when gravity is modified")
    plt.xlabel("Gravity")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig("./plots/mountain_car_vary_gravity.png")
    plt.show()  

plot_performance(results, ds='bg')


import ipdb; ipdb.set_trace()
