# DP+CVB+Eps: A Combined Epsilon Perturbation and Differential Privacy Approach

This repository contains the implementation of our proposed privacy-preserving framework for **Federated Learning (FL)**, combining Convolutional Variational Bottlenecks (CVB), epsilon perturbation, and Differential Privacy (DP) to defend against gradient inversion attacks.

Our work addresses the vulnerability of CVB to the Variational Bottleneck Inversion Attack (VBIA), which can exploit the reparameterization mechanism to reconstruct private training images from shared gradients. We propose a two-layer defense that perturbs attacker-injected latent variables during the CVB forward pass and sanitizes gradients before they are shared.

## Research Paper

**Title:** Defending Convolutional Variational Bottlenecks Against Gradient Inversion in Federated Learning: A Combined Epsilon Perturbation and Differential Privacy Approach

**Authors:** Maheen Fatima Rasool, Ahmed Zubair

**Affiliation:** Department of Artificial Intelligence, FAST-NUCES, Islamabad, Pakistan

**Paper:** [Read the Research Paper](DP-CVB-Eps-A-Combined-Epsilon-Perturbation-and-Differential-Privacy-Approach.pdf)

## Key Contributions

- **Epsilon Perturbation:** Introduces fresh Gaussian noise into attacker-injected reparameterization variables within the CVB forward pass.
- **Differential Privacy:** Applies gradient clipping and Gaussian noise to reduce information leakage through shared gradients.
- **Combined Defense:** Addresses both the latent reparameterization attack surface and gradient-level information leakage.
- **Gradient Inversion Evaluation:** Evaluates the proposed framework against VBIA and other gradient inversion attacks.
- **Privacy–Utility Analysis:** Evaluates reconstruction quality and classification accuracy to assess privacy protection and model utility.

## Project Demonstrations

The repository includes Jupyter notebooks demonstrating federated training and reconstruction attacks.

- **Federated Training:** Follow [`federated_training.ipynb`](federated_training.ipynb) for demonstrations of federated model training.
- **Reconstruction Attacks:** Follow [`reconstruction.ipynb`](reconstruction.ipynb) for demonstrations of VBIA and image reconstruction experiments.

## Methodology

The proposed framework combines two complementary defense mechanisms:

1. **CVB with Epsilon Perturbation:** Adds independently sampled Gaussian noise to attacker-injected reparameterization variables before the bottleneck sampling operation.
2. **Differentially Private Gradient Sanitization:** Clips gradient norms and adds Gaussian noise before gradient exchange.

Together, these mechanisms aim to prevent attackers from reliably optimizing reconstructed images and dummy latent variables against the victim's shared gradients.

## Dataset

Our experiments use the CIFAR-10 dataset.

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

We provide data-loading utilities for the experiments. For reconstruction experiments, victim datasets were randomly sampled from the training datasets. These victim datasets are available in `data/victim_datasets`.

## Requirements

The project uses the following Python libraries and frameworks:

- matplotlib
- seaborn
- munch
- dill
- prettytable
- pyyaml
- numpy
- pandas
- pytorch
- torchvision
- torchmetrics
- einops
- lpips

Install the dependencies using the project's dependency configuration, if available, or install the required packages in your Python environment.

## Experimental Configuration

The experiments use a simulated federated learning environment with the following configuration:

- **Federated Learning Algorithm:** FedAvg
- **Number of Clients:** 10
- **Communication Rounds:** 300
- **Gradient Clipping Threshold:** 20.0
- **DP Noise Multiplier:** 0.001
- **Evaluation Dataset:** CIFAR-10

Refer to the notebooks and implementation for the complete training and reconstruction workflows.

## Limitations and Future Work

Future work includes integrating Opacus for per-sample gradient clipping and formal differential privacy accounting, evaluating the defense against multi-sample attacks and multiple local training rounds, and extending the evaluation to larger architectures and more complex datasets.
