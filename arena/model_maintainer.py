"""
    Content: This class tracks a model's metadata
    Author: Yiyuan Yang
    Date: April 30th 2020
"""

from Evolution.model.models.resnet.arena_resnet import gen_model
import pickle
import torch
import os

class ModelMaintainer(object):
    def __init__(self, config=None, arena_save_dir=None, arena_id=None):
        if config:
            self.config = config
            self.config["device"] = "cuda"
            self.config["save_dir"] = os.path.join(self.config["save_dir"], str(self.config["model_id"]))
            self.config["accuracies"] = {0:{},1:{},2:{}}
            self.config["losses"] = {0:{},1:{},2:{}}
        else:
            self.load_config(arena_save_dir, arena_id)

    """
    Getter Functions
    """
    def model_id(self):
        return self.config["model_id"]

    def device(self):
        return self.config["device"]

    def arena_id(self):
        return self.config["arena_id"]

    def epoch(self):
        return self.config["epoch"]

    def lineage(self):
        return self.config["lineage"]
    
    def age_left(self):
        return self.config["age_left"]

    def shield(self):
        return self.config["shield"]
    
    def random_seed(self):
        return self.config["random_seed"]

    def original_learning_rate(self):
        return self.config["backprop_config"]["learning_rate"] 

    def learning_rate(self):
        return self.config["backprop_config"]["learning_rate"] * (self.config["backprop_config"]["gamma"]**self.epoch())

    def gradient_clip(self):
        return self.config["backprop_config"]["gradient_clip"]

    def gradient_clip_value(self):
        return self.config["backprop_config"]["gradient_clip_value"]

    def save_dir(self):
        return self.config["save_dir"]

    def model_save_dir(self, epoch = None):
        if epoch is None:
            epoch = self.epoch()
        return os.path.join(self.config["save_dir"], str(epoch) + "_model.pt")
    
    def optim_save_dir(self, epoch = None):
        if epoch is None:
            epoch = self.epoch()    
        return os.path.join(self.config["save_dir"], str(epoch) + "_optim.pt")

    def config_save_dir(self, epoch = None):
        if epoch is None:
            epoch = self.epoch()
        return os.path.join(self.config["save_dir"], str(epoch) + "_config.pickle")

    def arena_config_save_dir(self):
        return os.path.join(self.config["arena_save_dir"], str(self.arena_id()) + "_config.pickle")

    def accuracy(self, phase, epoch = None):
        if epoch is None:
            return self.config["accuracies"][phase]
        else:
            return self.config["accuracies"][phase][epoch]
    
    def loss(self, phase, epoch = None):
        if epoch is None:
            return self.config["losses"][phase]
        else:
            return self.config["losses"][phase][epoch]

    """
    Setters
    """
    def aging(self):
        assert self.config["age_left"] > 0, "Age Left Cannot go negative"
        self.config["age_left"] -=1

    def epoch_step(self):
        self.config["epoch"] +=1
        if self.config["shield"] > 0:
            self.config["shield"] -= 1

    def set_accuracy(self, phase, accuracy):
        self.config["accuracies"][phase][self.epoch()] = accuracy

    def set_loss(self, phase, loss):
        self.config["losses"][phase][self.epoch()] = loss

    """
    Loading And Saving
    """
    def load_config(self, arena_save_dir, arena_id):
        arena_config_save_dir = os.path.join(arena_save_dir, arena_id + "_config.pickle")
        with open(arena_config_save_dir, 'rb') as handle:
            self.config = pickle.load(handle)

    def init_model(self, ModelCandidate):
        ModelCandidate.model = gen_model(self.config["model_config"]).to(self.device())
        ModelCandidate.optim = torch.optim.Adam(
            ModelCandidate.model.parameters(),
            lr=self.learning_rate(),
            betas=(0.9,0.999)
        ) 

    def load_model(self, ModelCandidate, epoch = None):
        if epoch is None:
            epoch = self.epoch()
        self.init_model(ModelCandidate)
        if self.model_exists(ModelCandidate):
            ModelCandidate.model.load_state_dict(torch.load(self.model_save_dir(epoch)))
            ModelCandidate.optim.load_state_dict(torch.load(self.optim_save_dir(epoch)))

    def model_exists(self, ModelCandidate, epoch = None):
        if epoch is None:
            epoch = self.epoch()
        return os.path.exists(self.model_save_dir(epoch)) and os.path.exists(self.optim_save_dir(epoch))

    def save_config(self):
        with open(self.config_save_dir(), "wb") as handle: # Save a copy in the model's own folder
            pickle.dump(self.config, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.arena_config_save_dir(), "wb") as handle: # Save a copy in the arena
            pickle.dump(self.config, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_model(self, candidate):
        assert candidate.model is not None and candidate.optim is not None, "Cannot Save Non Existent Model and Optimizer"
        torch.save(candidate.model.state_dict(), self.model_save_dir())
        torch.save(candidate.optim.state_dict(), self.optim_save_dir())

    def unload_model(self, ModelCandidate):
        with torch.no_grad():
            del ModelCandidate.model
            del ModelCandidate.optim
            ModelCandidate.model = None
            ModelCandidate.optim = None

    """
    Prep for training
    """
    def epoch_prep(self, ModelCandidate, phase=0):
        if phase == 0:
            ModelCandidate.model.train()
        else:
            ModelCandidate.model.eval()


    """
        Logger Related
    """
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


    