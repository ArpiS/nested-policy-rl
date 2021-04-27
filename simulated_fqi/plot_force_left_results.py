import seaborn as sns
import matplotlib.pyplot as plt
import json

with open('force_left=0.json') as f:
  results = json.load(f)


fqi_results = []
cfqi_results = []
for alg in ['cfqi', 'fqi']:
    for key in results[alg]:
        if alg == 'fqi':
            fqi_results.extend(results[alg][key])
        else:
            cfqi_results.extend(results[alg][key])
sns.distplot(fqi_results, label='FQI', bins=30)
sns.distplot(cfqi_results, label='CFQI', bins=30)
plt.legend()
plt.xlabel("Steps survived")
plt.title("Force left = 0")
plt.show()


import ipdb; ipdb.set_trace()
