import json
from train_cnfqi import run, warm_start
import numpy as np

num_iter = 10000
results = {}
for i in range(11):
    results[i] = {}
    results[i]["cfqi"] = {}
    results[i]["fqi"] = {}
    results[i]["warm_start"] = {}
for i in range(num_iter):
    for f in range(11):
        print(str(i) + str(f))
        performance = run(
            verbose=False, is_contrastive=True, evaluations=2, force_left=f
        )
        results[f]["cfqi"][i] = performance
        performance = run(
            verbose=False, is_contrastive=False, evaluations=2, force_left=f
        )
        results[f]["fqi"][i] = performance
        performance = warm_start(
            verbose=False, is_contrastive=False, evaluations=2, force_left=0
        )
        results[f]["warm_start"][i] = performance
    with open("warm_start_forcerange.json", "w") as f:
        json.dump(results, f)
