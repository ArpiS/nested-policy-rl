import json
from train_cnfqi import run
import numpy as np
import scipy.stats


num_iter=10000
results = {}
for i in range(11):
    results[i] = {}
    results[i]['cfqi'] = {}
    results[i]['fqi'] = {}
for i in range(num_iter):
    for f in range(11):
        printed_bg, printed_fg, performance, nfq_agent, X, X_test = run(verbose=False, is_contrastive=True, evaluations=2, force_left=f)
        results[f]['cfqi'][i] = performance
        printed_bg, printed_fg, performance, nfq_agent, X, X_test = run(verbose=False, is_contrastive=False, evaluations=2, force_left=f)
        results[f]['fqi'][i] = performance
    with open('force_left_v_performance.json', 'w') as f:
        json.dump(results, f)  
        