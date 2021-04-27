import json
from train_cnfqi import run
num_iter=10000
results = {'fqi': {}, 'cfqi': {}}
for i in range(num_iter):
    printed_bg, printed_fg, performance, nfq_agent = run(verbose=False, is_contrastive=True, evaluations=2, force_left=0)
    results['cfqi'][i] = performance
    printed_bg, printed_fg, performance, nfq_agent = run(verbose=False, is_contrastive=False, evaluations=2, force_left=0)
    results['fqi'][i] = performance
    with open('force_left=0.json', 'w') as f:
        json.dump(results, f)