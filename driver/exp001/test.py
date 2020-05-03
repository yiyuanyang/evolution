
import sys
import os
import torch
project_path = "C:\\Users\\yangy\\Documents\\ComputerVision\\Projects\\code\\projects"
config_path = "Evolution\\config\\experiment_config\\exp001\\gradient_descent.yml"
sys.path.append(project_path)
from Evolution.train.exp001.experiment_preparer import ExperimentPreparer
from Evolution.train.exp001.gradient_descent import Trainer
from Evolution.data.CIFAR10.CIFAR10_dataset import CIFAR10Dataset
from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np
config_path = os.path.join(project_path, config_path)

# Used to play around with training data for CIFAR 10
def test_train_dataset_data_loader():
    exp_prep = ExperimentPreparer(config_path)
    _,data_config,_,_ = exp_prep.get_each_config()
    train_dataset = CIFAR10Dataset(data_config["train_data"], 32)
    train_loader = data.DataLoader(train_dataset, **data_config["data_loader_params"])
    return train_dataset, train_loader

def get_sample_images():
    _, train_loader = test_train_dataset_data_loader()
    data, label = None, None
    for index, (image, gt) in enumerate(train_loader):
        if index == 1:
            break
        data = image
        label = gt
    return np.moveaxis(data.data.numpy().tolist(),1,3), label.data.numpy().tolist()

def imshow(image):
    plt.figure()
    plt.imshow(image)

