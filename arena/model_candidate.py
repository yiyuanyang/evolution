"""
    Content: This is a class that stores and operates on a single model
    Author: Yiyuan Yang
    Date: April 30th 2020
"""

import os
import torch
from torch import nn
from Evolution.arena.model_maintainer import ModelMaintainer
from Evolution.utils.lineage.lineage_tree import Lineage
from Evolution.utils.log.logger import Logger
import pickle
import numpy as np

sample_model_candidate_initialization_config = {
    "arena_id": 10,
    "model_id": 101,
    "epoch": 45,
    "save_dir": "somewhere",
    "arena_save_dir": "somewhere else",
    "model_config": {},
    "backprop_config": {},
    "random_seed": 0,
    "age_left": 0,
    "shield_epoch": 13,
    "lineage": Lineage(0, None, None)
}


def gen_model_candidate_config(
    arena_id,
    model_id,
    epoch,
    save_dir,
    arena_save_dir,
    model_config,
    backprop_config,
    random_seed,
    age_left = 8,
    shield = 10,
    lineage = None
):
    if lineage is None:
        lineage = Lineage(model_id, None, None)
    return vars()


class ModelCandidate(object):
    def __init__(
        self,
        config
    ):
        self.mm = ModelMaintainer(config)
        self.device = torch.device(config["device"])
        if not os.path.exists(self.mm.save_dir()):
            os.system("mkdir " + self.mm.save_dir())
        self.logger = Logger(self.mm.save_dir())
        if not self.mm.model_exists(self):
            self.load_model()
            self.save()
            self.unload()

    def save_model(self):
        self.logger.log_model_activity("Saving Model", self)
        self.mm.save_model(self)

    def save_config(self):
        self.logger.log_model_activity("Saving Config", self)
        self.mm.save_config()

    def save(self):
        self.logger.log_model_activity("Saving Snapshot ", self)
        self.save_model()
        self.save_config()

    def unload(self):
        self.logger.log_model_activity("Unloading", self)
        self.mm.unload_model(self)

    def load_model(self):
        self.logger.log_model_activity("Loading Model", self)
        self.mm.load_model(self)

    def run_round(
        self,
        epoch,
        epoch_per_round,
        data_loaders
    ):
        assert (self.mm.epoch() - epoch) >= 0 and (self.mm.epoch() - epoch) < epoch_per_round, "Mismatch between model candidate epoch and arena epcoh" 
        """
            Load in the model, perform epochs_per_round epochs of backpropagation
            Save the model and return performance
        """
        self.logger.log_model_activity("Starting Round ", self)
        self.load_model()
        for i in range(epoch_per_round):
            while self.mm.model_exists(self, epoch + i + 1):
                continue
            self.run_epoch(data_loader=data_loaders[0], phase=0)
            self.run_epoch(data_loader=data_loaders[1], phase=1)
            self.run_epoch(data_loader=data_loaders[2], phase=2)
            self.mm.epoch_step()
            self.save()
        self.mm.aging()
        with torch.no_grad():
            self.unload()


    def run_epoch(
        self, 
        data_loader, 
        phase
    ):
        self.logger.log_model_activity("Starting Epoch {epoch} ".format(epoch=self.mm.epoch()), self)
        self.mm.epoch_prep(self, phase)
        self.logger.set_phase(self.mm.epoch(), phase)
        self.logger.log("Epoch: {epoch} for arena_id: {arena_id} and model_id: {model_id}".format(
            epoch=self.mm.epoch(),
            arena_id=self.mm.arena_id(),
            model_id=self.mm.model_id()
        ))
        predictions, ground_truths, losses = [],[],[]
        for batch_index, (data, ground_truth) in enumerate(data_loader):
            prediction_prob, prediction, loss = self.step(data, ground_truth, phase)
            predictions += prediction
            ground_truths += ground_truth.cpu().numpy().tolist()
            losses.append(loss)
            num_batches = len(data_loader)
            self.mm.batch_log(batch_index, num_batches, data, ground_truth, prediction_prob, prediction, loss, self.logger)
        accuracy, loss = self.logger.log_epoch_metrics(self.mm.epoch(), ground_truths, predictions, losses)
        self.mm.set_accuracy(phase, accuracy)
        self.mm.set_loss(phase, loss)
        if phase == 0:
            self.model.log_weights(self.logger)


    def step(self, data, ground_truth, phase):
        data, ground_truth = data.to(self.device).type(torch.float32), ground_truth.to(self.device)
        prediction = self.model(data)
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(prediction, ground_truth)
        self.optim.zero_grad()
        if phase == 0:
            loss.backward()
            if self.mm.gradient_clip():
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.mm.gradient_clip_value()
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
        self.load_model()
        other_candidate.load_model()
        torch.manual_seed(self.mm.random_seed())
        np.random.seed(self.mm.random_seed())
        self.model.breed_net(other_candidate.model, logger, mutation_policy, max_weight_mutation)
        new_model_save_dir = os.path.join(self.mm.save_dir(), str(new_model_id))
        torch.save(self.model.state_dict(), os.path.join(new_model_save_dir, str(self.mm.epoch()) + "_model.pt"))
        torch.save(self.optim.state_dict(), os.path.join(new_model_save_dir, str(self.mm.epoch()) + "_optim.pt"))
        self.logger.log_breed(self.mm.model_id(), other_candidate.model_id(), new_model_id)
        self.unload()
        other_candidate.unload()
