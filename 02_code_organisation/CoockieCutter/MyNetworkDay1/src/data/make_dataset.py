# -*- coding: utf-8 -*-
import torch
from torchvision import datasets, transforms

def mnist():
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
    # exchange with the real mnist dataset
    trainset = datasets.MNIST('../../data/', download=True, train=True, transform=transform)
    testset = datasets.MNIST('../../data/', download=True, train=False, transform=transform)
    return trainset, testset

mnist()