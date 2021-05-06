import seaborn as sns
import matplotlib.pyplot as plt
import json
import scipy
import numpy as np
import pandas as pd
import matplotlib
font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

##### Class imbalance

with open('class_imbalance_cfqi.json') as f:
  results = json.load(f)

results_df = pd.DataFrame(results)
results_df = pd.melt(results_df)

plt.figure(figsize=(7, 5))
sns.boxplot(data=results_df, x="variable", y="value", order=["fg_only", "fqi_joint", "cfqi"])
plt.xticks(np.arange(3), ["FQI (FG only)", "FQI (Joint)", "CFQI"])
plt.xlabel("")
plt.ylabel("Number of successful steps")
plt.tight_layout()
# plt.savefig("")
plt.show()
import ipdb; ipdb.set_trace()


#### Force=0

# with open('force_left=0.json') as f:
#   results = json.load(f)


# fqi_results = []
# cfqi_results = []
# for alg in ['cfqi', 'fqi']:
#     for key in results[alg]:
#         if alg == 'fqi':
#             fqi_results.extend(results[alg][key])
#         else:
#             cfqi_results.extend(results[alg][key])
# sns.distplot(fqi_results, label='FQI', bins=30)
# sns.distplot(cfqi_results, label='CFQI', bins=30)
# plt.legend()
# plt.xlabel("Steps survived")
# plt.title("Force left = 0")
# plt.show()




#### Range of forces

with open('force_left_v_performance.json') as f:
  results = json.load(f)

# import ipdb; ipdb.set_trace()

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

c_success = []
f_success = []
c_errs = []
f_errs = []
# for i in range(11):
# import ipdb; ipdb.set_trace()
for i in [0, 5, 10]:
    cfqi_perf = []
    fqi_perf = []
    for key in results[str(i)]['fqi']:
        fqi_perf.extend(results[str(i)]['fqi'][key])
    for key in results[str(i)]['cfqi']:
        cfqi_perf.extend(results[str(i)]['cfqi'][key])
    c_success.append(np.mean(cfqi_perf))
    f_success.append(np.mean(fqi_perf))
    m, h = mean_confidence_interval(cfqi_perf)
    c_errs.append(h)
    m, h = mean_confidence_interval(fqi_perf)
    f_errs.append(h)
# x = [k for k in range(11)]
x = [0, 5, 10]

sns.scatterplot(x, c_success, label='CFQI')
plt.errorbar(x, c_success ,yerr=c_errs, linestyle="None")
sns.scatterplot(x, f_success, label='FQI')
plt.errorbar(x, f_success ,yerr=f_errs, linestyle="None")
plt.title("Performance of CFQI and FQI when force on cart is modified")
plt.xlabel("Force Left")
plt.ylabel("Steps Survived")
plt.show()

fqi_results = []
cfqi_results = []
for alg in ['cfqi', 'fqi']:
    for key in results['0'][alg]:
        if alg == 'fqi':
            fqi_results.extend(results['0'][alg][key])
        else:
            cfqi_results.extend(results['0'][alg][key])
sns.distplot(fqi_results, label='FQI', bins=30)
sns.distplot(cfqi_results, label='CFQI', bins=30)
plt.legend()
plt.xlabel("Steps survived")
plt.title("Force left = 0")
plt.show()


import ipdb; ipdb.set_trace()
