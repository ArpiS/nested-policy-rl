import json
from train_cnfqi import run, warm_start, transfer_learning
num_iter = 10000
results = {'fqi': {}, 'cfqi': {}, 'warm_start': {}, 'tl': {}}
for i in range(num_iter):

    performance = run(verbose=False, is_contrastive=True, evaluations=2, force_left=0)
    results['cfqi'][i] = performance
    performance = run(verbose=False, is_contrastive=False, evaluations=2, force_left=0)
    results['fqi'][i] = performance
    performance = warm_start(verbose=False, is_contrastive=False, evaluations=2, force_left=0)
    results['warm_start'][i] = performance
    performance = transfer_learning(verbose=False, is_contrastive=False, evaluations=2, force_left=0)
    results['tl'][i] = performance
    
    
    with open('force_left=0.json', 'w') as f:
        json.dump(results, f)
