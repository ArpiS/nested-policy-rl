# Contrastive fitted Q-iteration: off-policy reinforcement learning for case-control environments

This repository is the official implementation of [Contrastive fitted Q-iteration: off-policy reinforcement learning for case-control environments](https://neurips.cc/).

Off-policy reinforcement learning (RL) has proven to be a powerful framework for guiding agents' actions in environments with stochastic rewards and unknown or noisy state dynamics. In many real-world settings, these agents are acting in distinct types of environments, each of whose dynamics differ slightly. For example, we may be interested in developing policies guiding medical treatment for patients with and without a given disease, or policies to navigate curriculum design for students with and without a learning disability. Here, we introduce contrastive fitted Q-iteration (CFQI), an off-policy RL framework that finds optimal policies in environments that exhibit this case-control structure. Our approach develops a contrastive Q-value function that leverages the shared structure between the two groups while allowing their policies to be distinct from one another. We find that CFQI yields a policy that relies on relevant features to make predictions; it also performs at least as well as a policy that does not consider group structure, and often much better because of a closer fit to the environments. Furthermore, CFQI is robust to imbalance in group sample sizes and outperforms other bespoke approaches for similar tasks, including warm start and transfer learning methods. We demonstrate the performance of CFQI using an OpenAI Gym environment and a clinical decision-making RL task.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

In this repository, we demonstrate how to reproduce results on an OpenAI gym environment. We include code but do not include instructions for preprocessing MIMIC-IV since it is a semi-private dataset available only to authorized users.

## Training

All of the algorithms in this paper are available in simulated_fqi/train.py. To run each of the algorithms (CFQI, FQI, Warm Start, Transfer Learning):

```train
python train.py fqi
python train.py cfqi
python train.py warm_start
python train.py transfer_learning
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

### Cartpole

CFQI can be evaluated and benchmarked against competing methods in several different settings in the Cartpole environment. Our experiments include testing:

- Overall cumulative reward as the leftward force varies (`simulated_experiments/test_force_range.py`)
- Interpretability using SHAP values (`simulated_experiments/notebooks/neurips_experiments.ipynb`)
- Performance when no difference exists between foreground and background (`simulated_experiments/shuffle_test.py`)
- Performance with sample size imbalance between the foreground and background (`simulated_experiments/sample_size_experiment.py`)

Each of these experiments corresponds to a `.py` file in the `simulated_experiments` directory. Each experiment can be run by calling its respective file. For example, to run the sample size experiment, one can run the following command from a terminal:

`python sample_size_experiment.py`

Each of these experiments outputs a `.json` file containing the results. The plots can be generated using the correspoding files in the `simulated_experiments/plotting` directory.

### MIMIC-IV

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).


## Results

![cartpole_results](simulated_fqi/plots/bg_force_v_performance.png)
<p float="left">
  <img src="simulated_fqi/plots/bg_force_v_performance.png" width="100" />
  <img src="simulated_fqi/plots/fg_force_v_performance.png" width="100" /> 
</p>

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it.


## Contributing

We include an MIT license. If you would like to contribute to this repository, create a branch and merge with a pull request to the original authors.
If you have questions, please contact Aishwarya Mandyam (aishwarya@princeton.edu).
