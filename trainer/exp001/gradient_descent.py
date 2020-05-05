"""
    Content: Training Driver File for gradient descent
    Author: Yiyuan Yang
    Date: April. 19th 2020
"""

import os
import torch
import torch.nn as nn
from torch.utils import data
from tqdm import tqdm
from Evolution.model.models.resnet.resnet import gen_model
from Evolution.data.CIFAR10.CIFAR10_dataset import CIFAR10Dataset


class Trainer(object):
    """
        This is for gradient descent
    """
    def __init__(self, experiment_preparer):
        self.device = torch.device("cuda")
        self.experiment_preparer = experiment_preparer
        self.basic_config, self.data_config, self.train_config, self.save_config = \
            self.experiment_preparer.get_each_config()
        self.logger = self.experiment_preparer.get_logger()
        self.model = gen_model(self.train_config["model_config"]).to(
            self.device)
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_config["learning_config"]["learning_rate"],
            betas=(0.9, 0.999))
        self.load_data()
        self.logger.log_config(
            config_name="Experiment Config",
            config=self.experiment_preparer.get_each_config())

    def load_data(self):
        augmentation_configs = self.data_config["augmentation_configs"]
        data_loader_params = self.data_config["data_loader_params"]

        train_dataset = CIFAR10Dataset(
            data_dir_list=self.data_config["train_data"],
            augmentation_config=augmentation_configs)
        eval_dataset = CIFAR10Dataset(
            data_dir_list=self.data_config["eval_data"],
            augmentation_config=None)
        test_dataset = CIFAR10Dataset(
            data_dir_list=self.data_config["test_data"],
            augmentation_config=None)

        train_loader = data.DataLoader(train_dataset, **data_loader_params)
        eval_loader = data.DataLoader(eval_dataset, **data_loader_params)
        test_loader = data.DataLoader(test_dataset, **data_loader_params)

        self.data_loaders = [train_loader, eval_loader, test_loader]

    def adjust_learning_rate(self, epoch):
        steps = self.train_config["learning_config"]["steps"]
        gamma = self.train_config["learning_config"]["gamma"]
        if epoch not in steps:
            self.logger.log("No need to adjust learning rate")
            return
        cur_learning_rate = self.learning_rate
        self.learning_rate *= gamma
        for param_group in self.optim.param_groups:
            param_group['lr'] = self.learning_rate

        self.logger.log_learning_rate_change(epoch=epoch,
                                             cur=cur_learning_rate,
                                             new=self.learning_rate)

    def train(self):
        # Get variables
        learning_config = self.train_config["learning_config"]
        self.learning_rate = learning_config["learning_rate"]

        self.logger.log_model(self.model)

        # Start training
        for epoch in tqdm(
                range(learning_config["start_epoch"],
                      learning_config["max_epoch"])):
            if learning_config["adjust_learning_rate"]:
                self.adjust_learning_rate(epoch)
            cur_model_name = "epoch_{}.pt".format(int(epoch))
            cur_model_path = os.path.join(self.save_config["model_save_dir"],
                                          "model_" + cur_model_name)
            cur_optim_path = os.path.join(self.save_config["model_save_dir"],
                                          "optim_" + cur_model_name)
            if os.path.exists(
                    cur_model_path) and learning_config["use_existing_model"]:
                self.model.load_state_dict(torch.load(cur_model_path))
                self.optim.load_state_dict(torch.load(cur_optim_path))
            else:
                self.epoch(epoch, 0)
                self.epoch(epoch, 1)
                self.epoch(epoch, 2)
                torch.save(self.model.state_dict(), cur_model_path)
                torch.save(self.optim.state_dict(), cur_optim_path)
            if learning_config["eval_only"]:
                self.epoch(epoch, 1)
                self.epoch(epoch, 2)

    def epoch(
        self,
        epoch,
        phase,
    ):
        """
            epoch: current epoch
            phase: 0 for training, 1 for eval, 2 for test
        """
        if phase == 0:
            self.model.train()
        else:
            self.model.eval()

        self.logger.set_phase(epoch, phase)

        all_prediction, all_ground_truth, all_loss = [], [], []
        for batch_index, (cur_data, ground_truth) in enumerate(self.data_loaders[phase]):
            # Some Logging data to see everything is correct
            if batch_index == 0 and phase != 2:
                self.logger.log_data(batch_index=batch_index,
                                     data=cur_data,
                                     label=ground_truth)
            if batch_index % 200 == 0:
                print("Batch: " + str(batch_index) + "/" + str(
                    len(self.data_loaders[phase]) //
                    self.data_config["data_loader_params"]["batch_size"]))

            # Calculations
            cur_data, ground_truth = cur_data.to(self.device).type(
                torch.float32), ground_truth.to(self.device)
            prediction = self.model(cur_data)
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(prediction, ground_truth)
            self.optim.zero_grad()
            if phase == 0:
                loss.backward()
                if self.train_config["learning_config"]["gradient_clip"]:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.train_config["learning_config"]
                        ["gradient_clipping"])
                self.optim.step()

            # Format
            prediction_prob = prediction.data.cpu().numpy().tolist()
            ground_truth = ground_truth.data.cpu().numpy().tolist()
            loss = loss.data.cpu().numpy().tolist()
            prediction = [pred.index(max(pred)) for pred in prediction_prob]
            all_prediction += prediction
            all_ground_truth += ground_truth
            all_loss.append(loss)

            # Logging the results
            if phase == 0 and batch_index % self.save_config[
                    "log_frequency"] == 0:
                self.logger.log_batch_result(batch_index=batch_index,
                                             num_batches=len(
                                                 self.data_loaders[phase]),
                                             prediction_prob=prediction_prob,
                                             prediction=prediction,
                                             ground_truth=ground_truth,
                                             loss=loss,
                                             top_n=8)

        # Record results from the entire epoch
        self.logger.log_epoch_metrics(epoch, all_ground_truth, all_prediction,
                                      all_loss)

        if phase == 0:
            self.model.log_weights(self.logger)
