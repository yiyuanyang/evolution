"""
    Content: This is a class that stores and operates on a single model
    Author: Yiyuan Yang
    Date: April 30th 2020
"""

import os
import torch
from torch import nn
from Evolution.utils.arena.model_config_manager import ModelConfigManager
from Evolution.utils.arena.model_file_manager import ModelFileManager
from Evolution.utils.lineage.lineage_tree import Lineage
from Evolution.utils.log.logger import Logger
import pickle
import numpy as np


class ModelCandidate(object):
    def __init__(
        self,
        config
    ):
        self.mcm = ModelConfigManager(config)
        self.mfm = ModelFileManager(config)
        self.logger = Logger(self.mcm.model_save_dir())

    def config(self):
        return self.mcm.get_config()

    def enter_arena(self, arena_id, epoch):
        # ** Only For New Model Candidates, Rather than loading Old Ones
        # ** The Epoch Should Be Same As The One It Replaces
        self.mcm.enter_arena(arena_id=arena_id, epoch=epoch)
        self.mfm.new_model_optimizer(self)
        self.mfm.save_snapshot(self)
        self.mfm.unload_model_optimizer(self)
        return self.config()

    def run_round(
        self,
        epoch,
        epoch_per_round,
        data_loaders
    ):
        # ** Load the model and perform one round of training
        self.logger.log_model_activity("Starting Round ", self)
        self.mfm.load_model_optimizer(self)
        for i in range(epoch_per_round):
            if self.mfm.model_exists(epoch + i) or \
                self.mfm.model_exists(epoch + epoch_per_round - 1):
                continue
            self.mcm.epoch_step()
            self.run_epoch(data_loader=data_loaders[0], phase=0)
            self.run_epoch(data_loader=data_loaders[1], phase=1)
            self.run_epoch(data_loader=data_loaders[2], phase=2)
            self.mfm.save_snapshot(self)
        self.mcm.aging()
        self.mfm.unload_model_optimizer(self)


    def epoch_prep(self, ModelCandidate, phase=0):
        if phase == 0:
            self.model.train()
        else:
            self.model.eval()


    def run_epoch(
        self, 
        data_loader, 
        phase
    ):
        # ** Prep, Variable Init, and Logging
        self.epoch_prep(phase)
        self.logger.log_model_activity("Starting Epoch {epoch} ".format(epoch=self.mcm.epoch()), self)
        self.logger.set_phase(self.mcm.epoch(), phase)
        self.logger.log("Epoch: {epoch} for arena_id: {arena_id} and model_id: {model_id}".format(
            epoch=self.mcm.epoch(),
            arena_id=self.mcm.arena_id(),
            model_id=self.mcm.model_id()
        ))
        predictions, ground_truths, losses = [],[],[]
        # ** Training
        for batch_index, (data, ground_truth) in enumerate(data_loader):
            prediction_prob, prediction, loss = self.step(data, ground_truth, phase)
            predictions += prediction
            ground_truths += ground_truth.cpu().numpy().tolist()
            losses.append(loss)
            num_batches = len(data_loader)
            self.batch_log(batch_index, num_batches, data, ground_truth, prediction_prob, prediction, loss, self.logger)
        accuracy, loss = self.logger.log_epoch_metrics(self.mcm.epoch(), ground_truths, predictions, losses)
        self.mcm.set_accuracy(phase, accuracy)
        self.mcm.set_loss(phase, loss)
        if phase == 0:
            self.model.log_weights(self.logger)

    def batch_log(self, batch_index, num_batches, data, ground_truth, prediction_prob, prediction, loss, logger):
        if batch_index % 100 == 0:
            logger.log_data(batch_index=batch_index, data=data, label=ground_truth)
            logger.log("Batch: " + str(batch_index) + "/" + str(num_batches))
            logger.log_batch_result(
                batch_index=batch_index,
                num_batches=num_batches,
                prediction_prob=prediction_prob,
                prediction=prediction,
                ground_truth=ground_truth.cpu().numpy().tolist(),
                loss=loss,
                top_n=8
            )

    def step(self, data, ground_truth, phase):
        data, ground_truth = data.to(self.mcm.device()).type(torch.float32), ground_truth.to(self.mcm.device())
        prediction = self.model(data)
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(prediction, ground_truth)
        self.optim.zero_grad()
        if phase == 0:
            loss.backward()
            if self.mcm.gradient_clip():
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.mcm.gradient_clip_value()
                )
            self.optim.step()
        prediction_prob = prediction.data.cpu().numpy().tolist()
        prediction = [pred.index(max(pred)) for pred in prediction_prob]
        ground_truth = ground_truth.data.cpu().numpy().tolist()
        loss = loss.data.cpu().numpy().tolist()
        return prediction_prob, prediction, loss


    def breed(
        self, 
        other_candidate, 
        new_arena_id,
        new_model_id, 
        logger,
        mutation_policy = "average",
        max_weight_mutation = 0.00005
    ):
        self.mfm.load_model_optimizer(self)
        other_candidate.load_model()
        torch.manual_seed(self.mcm.random_seed())
        np.random.seed(self.mcm.random_seed())
        self.model.breed_net(other_candidate.model, logger, mutation_policy, max_weight_mutation)
        new_model_save_dir = os.path.join(os.path.dirname(self.mcm.model_save_dir()), str(new_model_id))
        torch.save(self.model.state_dict(), os.path.join(new_model_save_dir, str(self.mcm.epoch()) + "_model.pt"))
        torch.save(self.optim.state_dict(), os.path.join(new_model_save_dir, str(self.mcm.epoch()) + "_optim.pt"))
        self.logger.log_breed(self.mcm.model_id(), other_candidate.mm.model_id(), new_model_id)
        self.mfm.unload_model_optimizer(self)
        other_candidate.mfm.unload_model_optimizer(self)
