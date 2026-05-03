# DP+CVB+Eps: A Combined Epsilon Perturbation and Differential Privacy Approach
This repository contains the implementation of our proposed Convolutional Variational Bottleneck (CVB).

For demonstrations of federated model training follow the federated_training.ipynb Jupyter notebook.
For demonstrations of Variational Bottleneck Inversion Attack (VBIA) follow the reconstruction.ipynb Jupyter notebook.

## Requirements:
+ matplotlib
+ seaborn
+ munch
+ dill
+ prettytable
+ pyyaml
+ numpy
+ pandas
+ pytorch=3.13.1
+ torchvision
+ torchmetrics=0.11.4
+ einops
+ lpips

## Dataset:
In our experiments we used CIFAR-10 dataset:
+ [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

We provide simple data loading utilities. 
For our reconstruction experiments, victim datasets were sampled randomly from the training datasets. 
We provide these very same victim datasets under *data/victim_datasets*.