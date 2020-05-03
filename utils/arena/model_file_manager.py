"""
    Content: This Class Manages the loading and saving of models and configs within its own folder
    Author: Yiyuan Yang
    Date: May 2nd 2020
"""

from Evolution.model.models.resnet.arena_resnet import gen_model
import pickle
import torch
import os

class ModelFileManager(object):
    def __init__(self, config):
        self.model_save_dir = config["model_save_dir"]

    def _model_dir(self, epoch):
        return os.path.join(self.model_save_dir, str(epoch) + "_model.pt") 


    def _optim_dir(self, epoch):
        return os.path.join(self.model_save_dir, str(epoch) + "_optim.pt")


    def _config_dir(self, epoch):
        return os.path.join(self.model_save_dir, "config.pt")

    def save_config(self, config):
        epoch = config["epoch"]
        with open(self._config_dir(epoch), "wb") as handle: # Save a copy in the model's own folder
            pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def model_exists(self, epoch):
        return os.path.exists(self._model_dir(epoch))

    def gen_model_optimizer(self, config):
        model = gen_model(config["model_config"]).to(config["device"])
        optim = torch.optim.Adam(
            model.parameters(),
            lr=config["backprop_config"]["learning_rate"],
            betas=(0.9,0.999)
        ) 
        return model, optim

    def new_model_optimizer(self, candidate):
        config = candidate.config()
        model, optim = self.gen_model_optimizer(config)
        candidate.model = model
        candidate.optim = optim

    def load_model_optimizer(self, candidate):
        config = candidate.config()
        epoch = config["epoch"]
        assert self.model_exists(epoch), "Model Does Not Exists In File"
        model, optim = self.gen_model_optimizer(config)
        model.load_state_dict(torch.load(self._model_dir(epoch)))
        optim.load_state_dict(torch.load(self._optim_dir(epoch)))
        candidate.model = model
        candidate.optim = optim
    
    def unload_model_optimizer(self, candidate):
        with torch.no_grad():
            model = candidate.model
            optim = candidate.optim
            del model
            del optim
            candidate.model = None
            candidate.optim = None

    def save_model_optimizer(self, candidate):
        config = candidate.config()
        epoch = config["epoch"]
        assert candidate.model is not None and candidate.optim is not None, "Cannot Save Non Existent Model and Optimizer"
        torch.save(candidate.model.state_dict(), self._model_dir(epoch))
        torch.save(candidate.optim.state_dict(), self._optim_dir(epoch))

    def save_snapshot(self, candidate):
        config = candidate.config()
        self.save_model_optimizer(candidate)
        self.save_config(config)
