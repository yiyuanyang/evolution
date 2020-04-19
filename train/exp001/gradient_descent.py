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
import tqdm
from model.models.resnet.resnet10_flatten import ResNet10_Flatten
from data.CIFAR10.CIFAR10_dataset import CIFAR10Dataset



class Trainer(object):

    def __init__(self, experiment_preparer):
        self.device = torch.device("cuda")
        self.experiment_preparer = experiment_preparer
        self.basic_config, self.data_config, self.train_config, self.save_config = \
            self.experiment_preparer.get_each_config()
        self.logger = self.experiment_preparer.get_logger()
        self.model = ResNet10_Flatten
        self.load_data

    def load_data(self):
        train_dataset = CIFAR10Dataset(
            self.data_config["train_data"],
            32
        )
        eval_dataset = CIFAR10Dataset(
            self.data_config["eval_data"],
            32
        )
        test_dataset = CIFAR10Dataset(
            self.data_config["test_data"],
            32
        )
        data_loader_params = self.data_config["data_loader_params"]
        self.train_loader = data.DataLoader(
            train_dataset, 
            **data_loader_params)
        self.eval_loader = data.DataLoader(
            eval_dataset,
            **data_loader_params
        )
        self.test_loader = data.DataLoader(
            test_dataset,
            **data_loader_params
        )
    
    def adjust_learning_rate(self, optimizer, epoch, steps, gamma):
        if epoch in steps:
            cur_lr = self.lr
            self.lr *= gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr
                self.logger.log(
                    "Epoch " + str(epoch) + " , adjusted learning rate from " + 
                    str(cur_lr) + " to " + str(self.lr)
                )

    def train(self):
        learning_config = self.train_config["learning_config"]
        for epoch in tqdm(range(
                learning_config["start_epoch"],
                learning_config["max_epoch"]
            )
        ):
            cur_model_name = "epoch_{}.pt".format(int(epoch))
            cur_model_path = os.path.join(
                self.save_config["model_save_dir"],
                cur_model_name
            )
            if os.path.exists(cur_model_path) and learning_config["use_existing_model"]:
                self.model.load_state_dict(torch.load(cur_model_path))
            else:
                self.train_epoch(epoch)
                torch.save(self.model.state_dict, cur_model_path)
            if learning_config["eval_only"]:
                self.eval_epoch(epoch)
                self.test_epoch(epoch)

    def epoch(
        self, 
        epoch,
        phase,
    ):
        """
            epoch: current epoch
            phase: 0 for training others for eval or test
        """
        if phase == 0:
            self.model.train()
        else
        pred, target, loss = [],[],[]
        for batch_index, data_target in enumerate(self.train_loader):







