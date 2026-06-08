import os
import random
import numpy as np

import torch
from torchvision import datasets, transforms

from models import *


# Set random seed
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# #Dataset configurations (mean, std, size, num_classes)
# #定义数据集为cifar10
# _dataset_name = ['cifar10']
#
# _mean = {
#     'cifar10':  [0.4914, 0.4822, 0.4465],
# }
#
# _std = {
#     'cifar10':  [0.2023, 0.1994, 0.2010],
# }
#
# _size = {
#     'cifar10':  (32, 32),
# }
#
# _num = {
#     'cifar10':  10,
# }
#
# #定义数据集为mnist
# _dataset_name = ['mnist']
#
# _mean = {
#     'mnist': [0.1307],  # MNIST数据集是灰度图像，均值仅有一个通道
# }
#
# _std = {
#     'mnist': [0.3081],  # 标准差也仅有一个通道
# }
#
# _size = {
#     'mnist': (28, 28),  # MNIST图像尺寸是28x28
# }
#
# _num = {
#     'mnist': 10,  # MNIST有10个类别（0-9）
# }

# Dataset configurations (mean, std, size, num_classes)
_dataset_config = {
    'cifar10': {
        '_mean': [0.4914, 0.4822, 0.4465],
        '_std': [0.2023, 0.1994, 0.2010],
        '_size': (32, 32),
        '_num': 10
    },
    'mnist': {
        '_mean': [0.1307],  # MNIST是灰度图像，只有一个通道
        '_std': [0.3081],
        '_size': (28, 28),
        '_num': 10
    }
}


_dataset_name = 'cifar10'  # 或 'mnist'


# def get_config(dataset):
#     assert dataset in _dataset_name, _dataset_name
#     config = {}
#     config['mean'] = _mean[dataset]
#     config['std']  = _std[dataset]
#     config['size'] = _size[dataset]
#     config['num_classes'] = _num[dataset]
#     return config

def get_config(dataset):
    assert dataset in _dataset_config, f"Dataset {dataset} is not defined in the configuration."
    config = {}
    config['mean'] = _dataset_config[_dataset_name]['_mean']
    config['std'] = _dataset_config[_dataset_name]['_std']
    config['size'] =_dataset_config[_dataset_name]['_size']
    config['num_classes'] = _dataset_config[_dataset_name]['_num']
    return config


def get_norm(dataset):
    assert dataset in _dataset_config, f"Dataset {dataset} is not defined in the configuration."

    # 从 _dataset_config 中提取均值和标准差
    mean = torch.FloatTensor(_dataset_config[dataset]['_mean'])
    std = torch.FloatTensor(_dataset_config[dataset]['_std'])

    # 定义标准化和反标准化
    normalize = transforms.Normalize(mean, std)
    unnormalize = transforms.Normalize(-mean / std, 1 / std)

    return normalize, unnormalize

# def get_norm(dataset):
#     assert dataset in _dataset_name, _dataset_name
#     mean = torch.FloatTensor(_mean[dataset])
#     std  = torch.FloatTensor(_std[dataset])
#     normalize   = transforms.Normalize(mean, std)
#     unnormalize = transforms.Normalize(- mean / std, 1 / std)
#     return normalize, unnormalize


# def get_transform(dataset, augment=False, tensor=False):
#     transforms_list = []
#     if augment:
#         transforms_list.append(transforms.Resize(_size[dataset]))
#         transforms_list.append(transforms.RandomCrop(_size[dataset], padding=4))
#
#         # Horizontal Flip
#         transforms_list.append(transforms.RandomHorizontalFlip())
#     else:
#         transforms_list.append(transforms.Resize(_size[dataset]))
#
#     # To Tensor
#     if not tensor:
#         transforms_list.append(transforms.ToTensor())
#
#     transform = transforms.Compose(transforms_list)
#     return transform


def get_augment(dataset):
    assert dataset in _dataset_config, f"Dataset {dataset} is not defined in the configuration."
    transforms_list = []
    transforms_list.append(transforms.RandomCrop(_dataset_config[dataset]['_size'], padding=4))
    transforms_list.append(transforms.RandomHorizontalFlip())
    transform = transforms.Compose(transforms_list)
    return transform


# #Get dataset
# def get_dataset(dataset, datadir='data', train=True, augment=True):
#     transform = get_transform(dataset, augment=train & augment)
#
#     # 加载 MNIST 数据集
#     if dataset == 'mnist':
#         transform = transforms.ToTensor()
#         dataset = datasets.MNIST(datadir, train=True, transform=transform, download=True)
#
#     # 遍历当前命名空间，找到 MNIST 数据集名称
#     for name, obj in globals().items():
#         if isinstance(obj, datasets.MNIST):
#             print("当前数据集名称为:", name)
#             break
#
#     # if dataset == 'cifar10':
#     #     dataset = datasets.CIFAR10(datadir, train, download=True, transform=transform)
#
#
#     return dataset


import torchvision.transforms as transforms
from torchvision import datasets


def get_transform(dataset, augment=True):
    if dataset == 'mnist':
        transform = [transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1))]  # 将单通道扩展为 3 通道
    elif dataset == 'cifar10':
        transform = [transforms.ToTensor()]
        if augment:
            transform = [
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(32, padding=4),
                        ] + transform
    else:
        raise ValueError("不支持的数据集名称")

    return transforms.Compose(transform)


def get_dataset(dataset, datadir='data', train=True, augment=True):
    transform = get_transform(dataset, augment=(train and augment))

    # 加载相应的数据集
    if dataset == 'mnist':
        dataset_instance = datasets.MNIST(datadir, train=train, transform=transform, download=True)
    elif dataset == 'cifar10':
        dataset_instance = datasets.CIFAR10(datadir, train=train, transform=transform, download=True)
    else:
        raise ValueError("不支持的数据集名称")

    return dataset_instance


# Get model
# def get_model(dataset, network):
#     print("Dataset:", dataset)  # 检查传入的数据集名称是否正确
#     print("_num dictionary:", _num)  # 检查 _num 字典的内容
#
#     if dataset not in _num:
#         raise KeyError(f"数据集 '{dataset}' 未在 _num 字典中定义")
#
#     num_classes = _num[dataset]
#
#     if network == 'resnet18':
#         model = resnet18(num_classes=num_classes)
#     elif network == 'resnet34':
#         model = resnet34(num_classes=num_classes)
#     elif network == 'vgg11':
#         model = vgg11(num_classes=num_classes)
#     elif network == 'vgg13':
#         model = vgg13(num_classes=num_classes)
#     else:
#         raise NotImplementedError
#
#     return model
def get_model(dataset, network):
    # 检查数据集是否存在于 _dataset_config 中
    assert dataset in _dataset_config, f"数据集 '{dataset}' 未在 _dataset_config 中定义"

    # 从 _dataset_config 中获取类别数
    num_classes = _dataset_config[dataset]['_num']

    # 打印数据集信息，用于调试
    print("Dataset:", dataset)
    print("Number of classes:", num_classes)

    # 根据指定的网络结构创建模型
    if network == 'resnet18':
        model = resnet18(num_classes=num_classes)
    elif network == 'resnet34':
        model = resnet34(num_classes=num_classes)
    elif network == 'vgg11':
        model = vgg11(num_classes=num_classes)
    elif network == 'vgg13':
        model = vgg13(num_classes=num_classes)
    else:
        raise NotImplementedError(f"网络结构 '{network}' 未实现")

    return model

