# Nested policy fitted Q-iteration: off-policy reinforcement learning for nested environments
![Build Status](https://github.com/bee-hive/contrastive-rl/actions/workflows/crl_workflow.yml/badge.svg)

This repository is the official implementation of [Nested Policy Reinforcement Learning](https://arxiv.org/abs/2110.02879).

Off-policy reinforcement learning (RL) has proven to be a powerful framework for guiding agents' actions in environments with stochastic rewards and unknown or noisy state dynamics. In many real-world settings, these agents must operate in multiple environments, each with slightly different dynamics. For example, we may be interested in developing policies to guide medical treatment for patients with and without a given disease, or policies to navigate curriculum design for students with and without a learning disability. Here, we introduce nested policy fitted Q-iteration (NFQI), an RL framework that finds optimal policies in environments that exhibit such a structure. Our approach develops a nested Q-value function that takes advantage of the shared structure between two groups of observations from two separate environments while allowing their policies to be distinct from one another. We find that NFQI yields policies that rely on relevant features and perform at least as well as a policy that does not consider group structure. We demonstrate NFQI's performance using an OpenAI Gym environment and a clinical decision making RL task. Our results suggest that NFQI can develop policies that are better suited to many real-world clinical environments.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

In this repository, we demonstrate how to reproduce results on an OpenAI gym environment. We include code but do not include instructions for preprocessing MIMIC-IV since it is a semi-private dataset available only to authorized users.

## Contributing

We include an MIT license. If you would like to contribute to this repository, create a branch and merge with a pull request to the original authors.
