

![Python 3.8](https://img.shields.io/badge/python-3.8-DodgerBlue.svg?style=plastic)
![Pytorch 2.3.1](https://img.shields.io/badge/pytorch-2.3.1-DodgerBlue.svg?style=plastic)
![Torchvision 0.18.1](https://img.shields.io/badge/torchvision-0.18.1-DodgerBlue.svg?style=plastic)
![CUDA 12.1](https://img.shields.io/badge/cuda-12.1-DodgerBlue.svg?style=plastic)
![License MIT](https://img.shields.io/badge/License-MIT-DodgerBlue.svg?style=plastic)

## Code Architecture

    ├── checkpoint        # Saved models
    ├── data              # Dataset folder
    ├── models            # Model architectures
    │   ├── resnet.py     # ResNet models
    │   └── vgg.py        # VGG models
    ├── dataset.py        # Dataset processing function
    ├── main.py           # Main function
    ├── partition.py      # (Implicit) partioning function
    ├── dpso.py           # optimization algorithm
    ├── train.py          # Training function
    ├── trigger_**.py     # Trigger function for various ablation experiment
    ├── trigger_wshape_all2.py   # Trigger function 
    └── utils.py          # Utility functions

## Experiments

We provide example code snippets for CIFAR-10 dataset. These can be easily plugged in and modified in `./utils.py`,
specifically within the `get_dataset(*)` functions.

### Usage

To train and evaluate a backdoored model using LOTUS, run:

```bash
python main.py --gpu 0
```

It involves three steps to launch LOTUS:

- Step 1 (Line 32): Train a clean model.
- Step 2 (Line 35): Train a surrogate model for partitioning.
- Step 3 (Line 38): Poison the model with backdoor triggers.

### Configurations

The specific arguments and hyperparameters used to launch LOTUS can be found in `./main.py`, particularly in lines
45-61.

| Hyperparameter | Default Value | Description                                                     |
|----------------|---------------|-----------------------------------------------------------------|
| gpu            | "0"           | GPU ID used to launch the experiment.                           |
| dataset        | "cifar10"     | The utilized dataset.                                           |
| network        | "resnet18"    | The utilized model architecture.                                |
| victim         | 0             | The attack victim label.                                        |
| target         | 9             | The attack target label.                                        |
| cluster        | "kmeans"      | Partitioning method.                                            |
| num_par        | 4             | Number of sub-partitions.                                       |
| n_indi         | 3             | Number of individual negative samples used in Trigger-focusing. |
| n_comb         | 1             | Number of combined negative samples used in Trigger-focusing.   |
| batch_size     | 128           | Batch size for training.                                        |
| epochs         | 100           | Total number of training epochs.                                |
| seed           | 1024          | Random seed for reproducibility.                                |

### Outputs

Several output files will be saved in the directory `./checkpoint`.

- `./checkpoint/clean.pt`: Clean model.
- `./checkpoint/lotus_best.pt`: Best backdoored model.
- `./checkpoint/lotus_final.pt`: Final backdoored model.
- `./checkpoint/result.json`: ASRs of different combinations of trigger and partition.
- `./checkpoint/surrogate.pt`: Surrogate model for partitioning.
- `./checkpoint/training.log`: Training logs.
