"""
    Content: Training Driver File for gradient descent
    Author: Yiyuan Yang
    Date: April. 19th 2020
"""

import os
import torch
import torch.nn as nn
from torch.utils import data
import numpy as numpy
from tqdm import tqdm
from Evolution.model.models.resnet.arena_resnet import gen_model
from Evolution.train.exp001.experiment_preparer import ExperimentPreparer
from Evolution.data.CIFAR10.CIFAR10_dataset import CIFAR10Dataset
from Evolution.arena.arena import Arena
import torch.nn.functional as F
from sklearn import metrics



class Trainer(object):
    """
        This is for arena
    """
    def __init__(self, config_path):

        self.device = torch.device("cuda")
        self.experiment_preparer = ExperimentPreparer(config_path)
        self.basic_config, self.data_config, self.train_config, self.save_config = \
            self.experiment_preparer.get_each_config()
        self.load_data()
        self.arena = Arena(self.data_loaders, self.train_config, self.save_config)
    

    def load_data(self):
        augmentation_config = self.data_config["augmentation_config"]
        data_loader_config = self.data_config["data_loader_config"]

        train_dataset = CIFAR10Dataset(
            data_dir_list=self.data_config["train_data"],
            augmentation_config=augmentation_config
        )
        eval_dataset = CIFAR10Dataset(
            data_dir_list=self.data_config["eval_data"],
            augmentation_config=None
        )
        test_dataset = CIFAR10Dataset(
            data_dir_list=self.data_config["test_data"],
            augmentation_config=None
        )

        train_loader = data.DataLoader(
            train_dataset, 
            **data_loader_config)
        eval_loader = data.DataLoader(
            eval_dataset,
            **data_loader_config)
        test_loader = data.DataLoader(
            test_dataset,
            **data_loader_config)

        self.data_loaders = [train_loader, eval_loader, test_loader]

    
    def start_experiment(self):
        self.arena.run_experiment()










