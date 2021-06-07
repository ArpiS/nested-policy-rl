import json
from train_cnfqi import run, warm_start

num_iter = 10000
results = {"fqi": {}, "cfqi": {}, "warm_start": {}}

for i in range(num_iter):
    print(str(i))
    performance = run(verbose=False, is_contrastive=True, evaluations=2, force_left=0)
    results["cfqi"][i] = performance
    performance = run(verbose=False, is_contrastive=False, evaluations=2, force_left=0)
    results["fqi"][i] = performance
    performance = warm_start(
        verbose=False, is_contrastive=False, evaluations=2, force_left=0
    )
    results["warm_start"][i] = performance
    with open("warm_start_force0.json", "w") as f:
        json.dump(results, f)
