"""
    Content: This class tracks a model's metadata
    Author: Yiyuan Yang
    Date: April 30th 2020
"""

import pickle
import os


class ModelConfigManager(object):
    def __init__(self, config):
        self.config = config
        if "device" not in self.config.keys():
            self.config["device"] = "cuda"
        if "accuracies" not in self.config.keys():
            self.init_stats()

    """
    Getter Functions
    """

    def get_config(self):
        return self.config

    def model_id(self):
        return self.config["model_id"]

    def device(self):
        return self.config["device"]

    def arena_id(self):
        assert "arena_id" in self.config.keys(), \
            "Arena ID Not Set For Current Candidate"
        return self.config["arena_id"]

    def epoch(self):
        assert "epoch" in self.config.keys(), \
            "Epoch Not Set For Current Candidate"
        return self.config["epoch"]

    def lineage(self):
        return self.config["lineage"]

    def age_left(self):
        return self.config["age_left"]

    def shield(self):
        return self.config["shield_epoch"]

    def random_seed(self):
        return self.config["random_seed"]

    def original_learning_rate(self):
        return self.config["backprop_config"]["learning_rate"]

    def learning_rate(self):
        return self.config["backprop_config"]["learning_rate"] * \
            (self.config["backprop_config"]["gamma"]**self.epoch())

    def gradient_clip(self):
        return self.config["backprop_config"]["gradient_clip"]

    def gradient_clip_value(self):
        return self.config["backprop_config"]["gradient_clip_value"]

    def model_save_dir(self):
        return self.config["model_save_dir"]

    def accuracy(self, phase):
        return self.config["accuracies"][phase]

    def cur_accuracy(self, phase):
        return self.config["accuracies"][phase][self.epoch()]

    def loss(self, phase):
        return self.config["losses"][phase]

    def cur_loss(self, phase):
        return self.config["losses"][phase][self.epoch()]

    """
    Setters
    """

    def aging(self):
        assert self.config["age_left"] > 0, \
            "Age Left Cannot go negative"
        self.config["age_left"] -= 1

    def epoch_step(self):
        self.config["epoch"] += 1
        if self.config["shield_epoch"] > 0:
            self.config["shield_epoch"] -= 1

    def init_stats(self):
        self.config["accuracies"] = {0: {}, 1: {}, 2: {}}
        self.config["losses"] = {0: {}, 1: {}, 2: {}}

    def set_accuracy(self, phase, accuracy):
        self.config["accuracies"][phase][self.epoch()] = accuracy

    def set_loss(self, phase, loss):
        self.config["losses"][phase][self.epoch()] = loss

    def enter_arena(self, arena_id, epoch):
        self.config["arena_id"] = arena_id
        self.config["epoch"] = epoch

    """
    Saving
    """

    def save_config(self):
        config_save_dir = os.path.join(self.model_save_dir(), "config.pickle")
        with open(config_save_dir, "wb") as handle:
            pickle.dump(self.config, handle, protocol=pickle.HIGHEST_PROTOCOL)
