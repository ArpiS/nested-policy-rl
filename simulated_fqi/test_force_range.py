import json
from train_cnfqi import run, warm_start, transfer_learning
import numpy as np
import scipy.stats

# num_iter=10000
# results = {}
# for i in range(11):
#     results[i] = {}
#     results[i]['cfqi'] = {}
#     results[i]['fqi'] = {}
#     results[i]['warm_start'] = {}
#     results[i]['tl'] = {}

# for i in range(num_iter):
#     for f in range(11):
#         print(str(i) + str(f))
#         performance = run(verbose=False, is_contrastive=True, evaluations=2, force_left=f)
#         results[f]['cfqi'][i] = performance
#         performance = run(verbose=False, is_contrastive=False, evaluations=2, force_left=f)
#         results[f]['fqi'][i] = performance
#         performance = warm_start(verbose=False, is_contrastive=False, evaluations=2, force_left=f)
#         results[f]['warm_start'][i] = performance
#         performance = transfer_learning(verbose=False, is_contrastive=False, evaluations=2, force_left=f)
#         results[f]['tl'][i] = performance
        
    
#     with open('force_left_v_performance.json', 'w') as f:
#         json.dump(results, f)



num_iter=10000
results = {}
for i in range(0, 11):
    results[i] = {}
    results[i]['cfqi'] = {}
    results[i]['fqi'] = {}
    results[i]['warm_start'] = {}
    results[i]['tl'] = {}
for i in range(num_iter):
    for f in range(0, 11):
        print(str(i) + str(f))
        performance_fg, performance_bg = run(verbose=False, is_contrastive=True, evaluations=2, force_left=f)
        results[f]['cfqi'][i] = (performance_fg, performance_bg)
        performance_fg, performance_bg = run(verbose=False, is_contrastive=False, evaluations=2, force_left=f)
        results[f]['fqi'][i] = (performance_fg, performance_bg)
        performance_fg, performance_bg = warm_start(verbose=False, is_contrastive=False, evaluations=2, force_left=f)
        results[f]['warm_start'][i] = (performance_fg, performance_bg)
        performance_fg, performance_bg = transfer_learning(verbose=False, is_contrastive=False, evaluations=2, force_left=f)
        results[f]['tl'][i] = (performance_fg, performance_bg)
    with open('force_left_v_performance.json', 'w') as f:
        json.dump(results, f)  


