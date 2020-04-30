"""
    Content: Defines the arena in which models compete
    Author: Yiyuan Yang
    Date: April 29th 2020
"""

import os
import torch
import torch.nn as nn
import Evolution.model.models.resnet.arena_resnet as arena_resnet
from Evolution.model.models.resnet.arena_resnet import gen_model
from sklearn import metrics
from Evolution.utils.lineage.lineage_tree import Lineage
import numpy as np
import pickle
import copy


sample_model_candidate_initialization_config = {
    "arena_id": 10,
    "model_id": 101,
    "epoch": 45,
    "device": "cuda",
    "save_dir": "somewhere",
    "arena_save_dir": "somewhere else",
    "age": 0,
    "shield_epoch": 13,
    "lineage": [None, None],
    "epoch_to_accuracy": {},
    "model_config": {},
    "backprop_config": {},
    "random_seed": 0
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
    age = 0,
    shield = 10,
    lineage = [None, None],
    device = "cuda",
    epoch_to_accuracy = {}
):
    return vars()


class ModelCandidate(object):
    def __init__(
        self,
        config
    ):
        self.init_from_config(config)


    def init_from_config(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        self.model = None
        self.optim = None
        self.learning_rate = self.config["backprop_config"]["learning_rate"]
        self.save_dir = os.path.join(self.config["save_dir"], self.config["model_id"])
        if not os.path.exists(self.save_dir):
            os.system("mkdir " + self.save_dir)
        self.arena_save_dir = self.config["arena_save_dir"]
        torch.manual_seed(self.config["random_seed"])
        np.random.seed(self.config["random_seed"])

    def save_model(self, epoch):
        assert self.model is not None and self.optim is not None, "Cannot Save Non Existent Model and Optimizer"
        model_save_dir = os.path.join(self.save_dir, str(epoch) + "_model.pt")
        optim_save_dir = os.path.join(self.save_dir, str(epoch) + "_optim.pt")
        torch.save(self.model.state_dict(), model_save_dir)
        torch.save(self.optim.state_dict(), optim_save_dir)

    def save_config(self, epoch):
        config_save_dir = os.path.join(self.save_dir, str(epoch) + "_config.pickle")
        arena_save_dir = os.path.join(self.arena_save_dir, str(self.config["arena_id"]) + "_config.pickle")

        with open(config_save_dir, "wb") as handle: # Save a copy in the model's own folder
            pickle.dump(self.config, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(arena_save_dir, "wb") as handle: # Save a copy in the arena
            pickle.dump(self.config, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_snapshot(self, epoch):
        self.save_model(epoch)
        self.save_config(epoch)

    def free_memory(self):
        with torch.no_grad():
            del self.model
            del self.optim
            self.model=None
            self.optim=None

    def protected(self):
        """
            Returns true if model cannot deprecated because of shielding
        """
        return self.config["shield_epoch"] > 0


    def init_model(self):
        self.model = gen_model(self.config["model_config"])
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["backprop_config"]["learning_rate"],
            betas=(0.9,0.999)
        )

    def load_model(self, load_optim=True):
        """
            require init_model() first
        """
        self.model.load_state_dict(torch.load(os.path.join(self.save_dir, str(self.config["epoch"] + "_model.pt"))))
        if load_optim:
            self.optim.load_state_dict(torch.load(os.path.join(self.save_dir, str(self.config["epoch"] + "_optim.pt"))))


    def epoch_prep(self, phase=0):
        if phase == 0:
            self.model.train()
        else:
            self.model.eval()
    
    def epoch(
        self, 
        epoch,
        data_loader, 
        phase, 
        logger, 
        loss_func = nn.CrossEntropyLoss()
    ):
        self.epoch_prep(phase)
        logger.set_phase(epoch, phase)
        logger.log("Epoch: {epoch} for arena_id: {arena_id} and model_id: {model_id}".format(
            epoch=epoch,
            arena_id=self.config["arena_id"],
            model_id=self.config["model_id"]
        ))
        predictions, ground_truths, losses = [],[],[]
        for batch_index, (data, ground_truth) in enumerate(data_loader):

            prediction_prob, prediction, ground_truth, loss = self.step(data, loss_func, ground_truth, phase)
            
            predictions.append(prediction)
            ground_truths.append(ground_truth)
            losses.append(loss)
            num_batches = len(data_loader) // data.shape[0]
            self.batch_log(batch_index, num_batches, data, ground_truth, prediction_prob, prediction, loss, logger)
        global_accuracy, _, _, _, _, _, _, _, _ = logger._log_epoch_metrics(epoch, ground_truths, predictions, losses)
        self.config["epoch_to_accuracy"][epoch] = global_accuracy
        self.config["shield"] -= 1
        if phase == 0:
            self.model.log_weights(logger)


    def adjust_learning_rate(
        self, 
        epoch,
        logger
    ):
        gamma = self.config["backprop_config"]["gamma"]
        cur_learning_rate = copy.deepcopy(self.learning_rate)
        self.learning_rate *= gamma
        for param_group in self.optim.param_groups:
            param_group['lr'] = self.learning_rate
        logger.log_learning_rate_change(
            epoch=epoch, 
            cur=cur_learning_rate, 
            new=self.learning_rate
        )


    def batch_log(self, batch_index, num_batches, data, ground_truth, prediction_prob, prediction, loss, logger):
        logger.log_data(batch_index=batch_index, data=data, ground_truth=ground_truth)
        if batch_index % 100 == 0:
            logger.log("Batch: " + str(batch_index) + "/" + str(num_batches))
            logger.log_batch_result(
                batch_index=batch_index,
                num_batches=num_batches,
                prediction_prob=prediction_prob,
                prediction=prediction,
                ground_truth=ground_truth,
                loss=loss,
                top_n=8
            )


    def step(self, data, ground_truth, loss_func, phase=0):
        data, ground_truth = data.to(self.device).type(torch.float32), ground_truth.to(self.device)
        prediction = self.model(data)
        loss = loss_func(prediction, ground_truth)
        self.optim.zero_grad()
        if phase == 0:
            loss.backward()
            if self.config["backprop_config"]["gradient_clip"]:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config["backprop_config"]["gradient_clip_value"]
                )
            self.optim.step()
        prediction_prob = prediction.data.cpu().numpy().tolist()
        prediction = [pred.index(max(pred)) for pred in prediction_prob]
        ground_truth = ground_truth.data.cpu().numpy().tolist()
        loss = loss.data.cpu().numpy().tolist()
        return prediction_prob, prediction, ground_truth, loss


    def breed(
        self, 
        other_candidate, 
        epoch,
        new_model_id, 
        save_path, 
        logger,
        mutation_policy = "average",
        max_weight_deviation = 0.05,
    ):
        self.model.breed_net(other_candidate.model, mutation_policy, max_weight_deviation)
        torch.save(self.model.state_dict(), os.path.join(save_path, str(epoch) + "_model.pt"))
        torch.save(self.optim.state_dict(), os.path.join(save_path), str(epoch) + "_optim.pt")
        logger.log_breed(self.model_id(), other_candidate.model_id(), new_model_id)


    def arena_id(self):
        return self.config["arena_id"]
    
    def model_id(self):
        return self.config["model_id"]

    def age(self):
        return self.config["age"]

    def shield(self):
        return self.config["shield"]
    
    def accuracy(self, epoch):
        return self.config["epoch_to_accuracy"][epoch]

    def lineage(self):
        return self.config["lineage"]

    def set_model(self, model, optim, learning_rate):
        self.model = model
        self.optim = optim
        self.learning_rate 

        

class Arena(object):
    def __init__(self, data_loaders, train_config, save_config):
        self.data_loaders = data_loaders
        self.train_config = train_config
        self.save_config = save_config
        self.model_config = train_config["evolution_config"]
        self.backprop_config = train_config["backprop_config"]
        self.evolution_config = train_config["evolution_config"]
        self.init_evolution_config(self.evolution_config)
        self.model_candidates = {}
        self.model_candidate_config_paths = {}
        self.model_save_dir = os.path.join(self.save_config["model_save_dir"], "all_models")
        self.arena_save_dir = os.path.join(self.save_config["model_save_dir"], "arena")
        if not os.path.exists(self.model_save_dir):
            os.system("mkdir " + self.model_save_dir)
            os.system("mkdir " + self.arena_save_dir)
        for i in range(self.evolution_config["num_models"]):
            cur_config_path = os.path.join(self.arena_save_dir, str(i) + "_config.pickle")
            self.model_candidate_config_paths[i] = cur_config_path
            if self.evolution_config["use_existing_model"]:
                with open(cur_config_path, 'rb') as handle:
                    cur_config = pickle.load(handle)
                    self.model_candidates[i] = ModelCandidate(cur_config)
                    self.new_model_id = self.evolution_config["cur_model_id"]
            else:
                cur_config = gen_model_candidate_config(
                    arena_id=i,
                    model_id=i,
                    epoch=self.evolution_config["start_epoch"],
                    save_dir=self.model_save_dir,
                    arena_save_dir=self.arena_save_dir,
                    model_config=self.train_config["model_config"],
                    backprop_config=self.train_config["backprop_config"],
                    random_seed=self.evolution_config["random_seed"] + i,
                    age = 0,
                    shield = self.evolution_config["max_shield_per_model"],
                    lineage = [None, None],
                    device = "cuda",
                    epoch_to_accuracy = {}
                )
                self.model_candidates[i] = ModelCandidate(cur_config)
                self.new_model_id = self.evolution_config["num_models"]

    def gen_new_model_id(self):
        model_id = self.new_model_id
        self.new_model_id += 1
        return model_id

    def end_of_round_update(self):
        survived = self.eliminate()
        self.breed(survived)

    def get_model_ids(self):
        return [model_candidate.model_id() for model_candidate in self.model_candidates]

    def get_model_ages(self):
        return [model_candidate.age() for model_candidate in self.model_candidates]

    def get_model_shields(self):
        return [model_candidate.shield()>0 for model_candidate in self.model_candidates]

    def get_performance(self):
        return [model_candidate.accuracy() for model_candidate in self.model_candidates]

    def get_open_arena_ids(self):
        return [arena_id for arena_id in range(self.num_models) if self.model_candidates[arena_id] is None]

    def get_open_arena_count(self):
        return len(self.get_open_arena_ids())

    def init_evolution_config(self, evolution_config):
        # Evolution Settings
        self.max_weight_deviation = evolution_config["max_weight_deviation"]
        self.max_round_per_model = evolution_config["max_round_per_model"]
        self.epoch_per_round = evolution_config["epoch_per_round"]
        self.num_models = evolution_config["num_models"],
        self.load_from_models = evolution_config["load_from_models"]
        self.elimination_rate = evolution_config["elimination_rate"]
        self.mutation_policy = evolution_config["mutation_policy"]
        self.random_seed = evolution_config["random_seed"]
        self.shield = evolution_config["shield"]
        self.max_weight_deviation = evolution_config["max_weight_deviation"]
        self.breed_intention_rate = evolution_config["breed_intention_rate"]
        self.downward_acceptance = evolution_config["downward_acceptance"]
        self.max_rounds = evolution_config["max_rounds"]

        self.round = evolution_config["start_round"]
        self.epoch = evolution_config["start_epoch"]
        self.round = self.epoch // self.epoch_per_round + 1
        self.cur_arena_id = evolution_config["cur_arena_id"]

    def eliminate(self):
        performance = self.get_performance()
        ages = self.get_model_ages()
        shields = self.get_model_shields()
        performance_list = [value for key, value in performance.items()].sort()
        survival_performance = performance_list[int(len(performance_list) * self.elimination_rate) + 1]
        eliminated = [
            arena_id 
            for arena_id, accuracy in performance.items() 
            if accuracy < survival_performance # Check if it performs badly
            and ages[arena_id] < self.max_round_per_model # Check if it needs to be deprecated
            and shields[arena_id] # Check if its protected    
        ]
        survived = [arena_id for arena_id in performance.keys() if arena_id not in eliminated]
        for arena_id in eliminated:
            cur_candidate = self.model_candidates[arena_id]
            self.model_candidates[arena_id] = None
            del cur_candidate
        return survived
    
    def breed(self, survived):
        performance = self.get_cur_performance()
        # TODO: Save Model Before They Are Trained
        # TODO: Write load model functions 


    def _breed(
        self, 
        parent_arena_id_1, 
        parent_arena_id_2, 
        target_arena_id,
        logger
    ):
        self.model_candidates[parent_arena_id_1].init_model()
        self.model_candidates[parent_arena_id_1].load_model()
        self.model_candidates[parent_arena_id_2].init_model()
        self.model_candidates[parent_arena_id_2].load_model()
        new_model_id = self.gen_new_model_id()
        new_lineage = Lineage(
            new_model_id, 
            self.model_candidates[parent_arena_id_1].lineage(),
            self.model_candidates[parent_arena_id_2].lineage()
        )
        new_config=gen_model_candidate_config(
            arena_id=target_arena_id,
            model_id=self.new_model_id,
            epoch=self.epoch,
            save_dir=self.model_save_dir,
            arena_save_dir=self.arena_save_dir,
            model_config=self.model_config,
            backprop_config=self.backprop_config,
            random_seed=self.random_seed + new_model_id,
            age=0,
            shield=self.shield,
            lineage = new_lineage,
            device = "cuda",
            epoch_to_accuracy={}
        )
        assert self.model_candidates[target_arena_id] is None, "Cannot Create A New Model At A Non Null Location"
        self.model_candidates[target_arena_id] = ModelCandidate(new_config)
        self.model_candidates[target_arena_id].save_config()
        self.model_candidates[target_arena_id].breed(
            other_candidate = self.model_candidates[parent_arena_id_2], 
            epoch = self.epoch,
            new_model_id = new_model_id, 
            save_dir = os.path.join(self.model_save_dir, new_model_id), 
            logger = logger,
            mutation_policy = "average",
            max_weight_deviation = 0.05,
        )
        # Now, all the state_dict are saved properly, just need to load them
        self.model_candidates[parent_arena_id_1].free_memory()
        self.model_candidates[parent_arena_id_2].free_memory()

        


    


