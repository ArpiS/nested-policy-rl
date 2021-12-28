import seaborn as sns
import matplotlib.pyplot as plt
import json
import scipy
import numpy as np
import pandas as pd
import matplotlib
from plottify import autosize

font = {"size": 20}
matplotlib.rc("font", **font)
# matplotlib.rcParams["text.usetex"] = True


# #### Force=0

# with open('force_left=0.json') as f:
#   results = json.load(f)

# import ipdb; ipdb.set_trace()
# fqi_results = []
# cfqi_results = []
# ws_results = []
# tl_results = []
# for alg in ['cfqi', 'fqi', 'warm_start', 'tl']:
#     for key in results[alg]:
#         if alg == 'fqi':
#             fqi_results.extend(results[alg][key])
#         elif alg == 'cfqi':
#             cfqi_results.extend(results[alg][key])
#         elif alg == 'warm_start':
#             ws_results.extend(results[alg][key])
#         else:
#             tl_results.extend(results[alg][key])

# plt.figure(figsize=(7, 5))

# sns.distplot(fqi_results, label='FQI')
# sns.distplot(cfqi_results, label='CFQI')
# sns.distplot(ws_results, label='Warm Start')
# sns.distplot(tl_results, label='Transfer Learning')
# plt.legend()
# plt.xlabel("Steps survived")
# plt.title("Force left = 0")
# plt.tight_layout()
# plt.savefig("./plots/forceleft0.png")
# plt.show()


with open("../force_left_v_performance.json") as f:
    results = json.load(f)

with open("../linear_model_force_range.json") as f:
    results_linear = json.load(f)
    
with open("force_left_v_performance_gfqi.json") as f:
    results_gfqi = json.load(f)

# import ipdb; ipdb.set_trace()
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, h


def plot_performance(results, ds="bg", figname=""):
    c_success = []
    f_success = []
    w_success = []
    t_success = []
    l_success = []
    g_success = []
    c_errs = []
    f_errs = []
    w_errs = []
    t_errs = []
    l_errs = []
    g_errs = []
    if ds == "bg":
        ind = 1
    else:
        ind = 0
    for i in range(0, 11):
        cfqi_perf = []
        fqi_perf = []
        ws_perf = []
        tl_perf = []
        linear_perf = []
        gfqi_perf = []
        for key in results[str(i)]["fqi"]:
            fqi_perf.extend(results[str(i)]["fqi"][key][ind])
        for key in results[str(i)]["cfqi"]:
            cfqi_perf.extend(results[str(i)]["cfqi"][key][ind])
        for key in results[str(i)]["warm_start"]:
            ws_perf.extend(results[str(i)]["warm_start"][key][ind])
        for key in results[str(i)]["tl"]:
            tl_perf.extend(results[str(i)]["tl"][key][ind])
        for key in results[str(i)]["cfqi"]:
            linear_perf.extend(results_linear[str(i)]["cfqi"][key][ind])
        for key in results_gfqi[str(i)]["gfqi"]:
            gfqi_perf.extend(results_gfqi[str(i)]["gfqi"][key][ind])
        c_success.append(np.mean(cfqi_perf))
        f_success.append(np.mean(fqi_perf))
        w_success.append(np.mean(ws_perf))
        t_success.append(np.mean(tl_perf))
        l_success.append(np.mean(linear_perf))
        g_success.append(np.mean(gfqi_perf))
        m, h = mean_confidence_interval(cfqi_perf)
        c_errs.append(h)
        m, h = mean_confidence_interval(fqi_perf)
        f_errs.append(h)
        m, h = mean_confidence_interval(ws_perf)
        w_errs.append(h)
        m, h = mean_confidence_interval(tl_perf)
        t_errs.append(h)
        m, h = mean_confidence_interval(linear_perf)
        l_errs.append(h)
        m, h = mean_confidence_interval(gfqi_perf)
        g_errs.append(h)
    x = [k for k in range(0, 11)]
    plt.figure(figsize=(10, 6))
    x_nfqi = np.add(x, 0.05)
    sns.scatterplot(x_nfqi, c_success, label="NFQI")
    plt.errorbar(x_nfqi, c_success, yerr=c_errs, linestyle="None")
    
    x_fqi = np.add(x, 0.1)
    sns.scatterplot(x_fqi, f_success, label="FQI")
    plt.errorbar(x_fqi, f_success, yerr=f_errs, linestyle="None")
    
    x_ws = np.add(x, 0.15)
    sns.scatterplot(x_ws, w_success, label="Warm Start")
    plt.errorbar(x_ws, w_success, yerr=w_errs, linestyle="None")
    
    x_tl = np.add(x, 0.2)
    sns.scatterplot(x_tl, t_success, label="Transfer Learning")
    plt.errorbar(x_tl, t_success, yerr=t_errs, linestyle="None")
    
    x_l = np.add(x, 0.25)
    sns.scatterplot(x_l, l_success, label="Linear NFQI")
    plt.errorbar(x_l, l_success, yerr=l_errs, linestyle="None")
    
    x_gfqi = np.add(x, 0.3)
    sns.scatterplot(x_gfqi, g_success, label="FQI with access to group label")
    plt.errorbar(x_gfqi, g_success, yerr=g_errs, linestyle="None")
    
    #plt.legend(prop={"size": 12})
    if ds == "bg":
        # plt.title("Background Dataset: Performance of CFQI, FQI, Warm Start, Transfer Learning when force on cart is modified")
        plt.title("Cartpole performance, background")
    else:
        plt.title("Cartpole performance, foreground")
    plt.xlabel("Force Left")
    plt.ylabel("Steps Survived")
    autosize()
    plt.tight_layout()
    plt.savefig(figname)
    plt.show()


# plt.figure(figsize=(24, 8))
# plt.suptitle("Performance when force on cart is modified")
# plt.subplot(121)
plot_performance(results, ds="bg", figname="bg_force_v_performance_linear_gfqi.png")
# plt.subplot(122)
plot_performance(results, ds="fg", figname="fg_force_v_performance_linear_gfqi.png")
# plt.tight_layout()
# plt.savefig("./plots/force_v_performance.png")
# plt.show()


import ipdb

ipdb.set_trace()


#### Range of forces

# with open('force_left_v_performance.json') as f:
#   results = json.load(f)

# def mean_confidence_interval(data, confidence=0.95):
#     a = 1.0 * np.array(data)
#     n = len(a)
#     m, se = np.mean(a), scipy.stats.sem(a)
#     h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
#     return m, h


# c_success = []
# f_success = []
# w_success = []
# t_success = []
# c_errs = []
# f_errs = []
# w_errs = []
# t_errs = []
# for i in range(11):
#     cfqi_perf = []
#     fqi_perf = []
#     ws_perf = []
#     tl_perf = []
#     for key in results[str(i)]['fqi']:
#         fqi_perf.extend(results[str(i)]['fqi'][key])
#     for key in results[str(i)]['cfqi']:
#         cfqi_perf.extend(results[str(i)]['cfqi'][key])
#     for key in results[str(i)]['warm_start']:
#         ws_perf.extend(results[str(i)]['warm_start'][key])
#     for key in results[str(i)]['tl']:
#         tl_perf.extend(results[str(i)]['tl'][key])

#     c_success.append(np.mean(cfqi_perf))
#     f_success.append(np.mean(fqi_perf))
#     w_success.append(np.mean(ws_perf))
#     t_success.append(np.mean(tl_perf))
#     m, h = mean_confidence_interval(cfqi_perf)
#     c_errs.append(h)
#     m, h = mean_confidence_interval(fqi_perf)
#     f_errs.append(h)
#     m, h = mean_confidence_interval(ws_perf)
#     w_errs.append(h)
#     m, h = mean_confidence_interval(tl_perf)
#     t_errs.append(h)

# x = [k for k in range(11)]
# plt.figure(figsize=(10, 7))
# sns.scatterplot(x, c_success, label='CFQI')
# plt.errorbar(x, c_success ,yerr=c_errs, linestyle="None")
# sns.scatterplot(x, f_success, label='FQI')
# plt.errorbar(x, f_success ,yerr=f_errs, linestyle="None")
# sns.scatterplot(x, w_success, label='Warm Start')
# plt.errorbar(x, w_success ,yerr=w_errs, linestyle="None")
# sns.scatterplot(x, t_success, label='Transfer Learning')
# plt.errorbar(x, t_success ,yerr=t_errs, linestyle="None")
# plt.title("Case-control environments")
# plt.xlabel("Force Left")
# plt.ylabel("Steps Survived")
# plt.tight_layout()
# plt.savefig("./plots/forceleftrange.png")
# plt.show()
