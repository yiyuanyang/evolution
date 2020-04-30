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
from Evolution.utils.lineage.lineage_tree import Lineage
import numpy as np


class ModelCandidate(object):
    def __init__(
        self,
        arena_id,
        model_id,
        epoch,
        save_dir,
        model_config,
        age = 0,
        shield_epoch = 20,
        lineage = [None, None]
    ):
        self.device = torch.device("cuda")
        self.model = None
        self.optim = None
        self.arena_id = arena_id
        self.model_id = model_id

        self.save_dir = os.path.join(save_dir, "model_id")
        self.age = age # Once reach an age, the model gets deprecated
        self.shield_epoch = shield_epoch # The model gets time to train for a while without needing to worry about competition
        self.lineage = Lineage(self.model_id, lineage[0], lineage[1])
        self.epoch_to_accurac = {}


    def save_snapshot(self, epoch):
        assert self.model is not None and self.optim is not None, "Cannot Save Non Existent Model and Optimizer"
        model_save_dir = os.path.join(self.save_dir, str(epoch) + "_model.pt")
        optim_save_dir = os.path.join(self.save_dir, str(epoch) + "_optim.pt")
        torch.save(self.model.state_dict(), model_save_dir)
        torch.save(self.optim.state_dict(), optim_save_dir)


    def protected(self):
        """
            Returns true if model cannot deprecated because of shielding
        """
        return self.shield_epoch > 0


    def init_model(self, model_config, backprop_config):
        self.model_config = model_config
        self.backprop_config = backprop_config
        self.model = gen_model(self.model_config)
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.backprop_config["learning_rate"],
            betas=(0.9,0.999)
        )

    
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
            arena_id=self.arena_id,
            model_id=self.model_id
        ))
        predictions, ground_truths, losses = [],[],[]
        for batch_index, (data, ground_truth) in enumerate(data_loader):
            num_batches = len(data_loader) // data.shape[0]
            prediction_prob, prediction, ground_truth, loss = self.step(data, loss_func, phase)
            predictions.append(prediction)
            ground_truths.append(ground_truth)
            losses.append(loss)
            self.epoch_log(batch_index, num_batches, data, ground_truth, prediction_prob, prediction, loss, logger)
        if phase == 0:
            self.model.log_weights(logger)


    def epoch_log(self, batch_index, num_batches, data, ground_truth, prediction_prob, prediction, loss, logger):
        logger.log_data(batch_index=batch_index, data=data, ground_truth=ground_truth)
        if batch_index % 100 == 0:
            logger.log("Batch: " + str(batch_index) + "/" + str(num_batches))
            logger.log_batch_result(
                batch_index=batch_index, 
                total_batches=num_batches, 
                prediction_prob=prediction_prob,
                prediction=prediction,
                ground_truth=ground_truth,
                loss=loss,
                top_n=8
            )


    def step(self, data, loss_func, phase=0):
        data, ground_truth = data.to(self.device).type(torch.float32), ground_truth.to(self.device)
        prediction = self.model(data)
        loss = loss_func(prediction, ground_truth)
        self.optim.zero_grad()
        if phase == 0:
            loss.backward()
            if self.backprop_config["gradient_clip"]:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.backprop_config["gradient_clip_value"]
                )
            self.optim.step()
        prediction_prob = prediction.data.cpu().numpy().tolist()
        prediction = [pred.index(max(pred)) for pred in prediction_prob]
        ground_truth = ground_truth.data.cpu().numpy().tolist()
        loss = loss.data.cpu().numpy().tolist()
        return prediction_prob, prediction, ground_truth, loss





class Arena(object):
    def __init__(self, train_config, save_config):
        self.init_record()
        evolution_config = train_config["evolution_config"]
        self.init_evolution_config(evolution_config)
        self.init_main_model(train_config)
        self.init_secondary_model(train_config)
        self.init_model_configs(save_config)
        # Track Each Model
        self.new_model_id = self.num_models


    def init_model_configs(self, save_config):
        self.save_path = save_config["model_save_dir"]
        for i in range(self.num_models):
            self.arena_id_to_model_id[i] = i
            cur_save_path = os.path.join(self.save_path, str(i))
            os.system("mkdir " + cur_save_path)
            self.model_id_to_per_epoch_model_path[i] = {1: cur_save_path}
            self.model_id_to_model_status_config[i] = arena_resnet.gen_model_status_config(
                initial_model=True,
                model_id=i,
                arena_id=i,
                lineage=[None, None],
                age=0
            )


    def gen_new_model_id(self):
        model_id = self.new_model_id
        self.new_model_id += 1
        return model_id


    def init_main_model(self, train_config):
        self.main_model = gen_model(train_config["model_config"]).to('cuda')
        self.optim = torch.optim.Adam(
            self.main_model.parameters(),
            lr=train_config["learning_config"]["learning_rate"],
            betas=(0.9,0.999)
        )


    def init_secondary_model(self, train_config):
        """
            This model is for breeding purposes
        """
        self.secondary_model = gen_model(train_config["model_config"]).to('cuda')


    def init_record(self):
        self.model_id_to_per_epoch_accuracy = {}
        # Example {0:{1:0.54, 2: 0.78},1:{1:0.65, 2: 0.67},2:{1: 0.34, 2: 0.70}}

        self.model_id_to_per_epoch_model_path = {}
        self.model_id_to_per_epoch_optim_path = {}
        # Example {0: {1:"abc/model", 2:"abc/model_2"}, 1: {1:"bcd/model", 2:"abc/model_2"}}

        self.model_id_to_model_status_config = {}

        self.arena_id_to_model_id = {}

    
    def end_of_epoch_update(self, accuracy):
        model_id = self.arena_id_to_model_id[self.cur_arena_id]
        if model_id not in self.model_id_to_per_epoch_model_path.keys():
            self.model_id_to_per_epoch_model_path[model_id] = {}
            self.model_id_to_per_epoch_accuracy[model_id] = {}
        cur_model_path = os.path.join(self.save_path, str(model_id), "model.pt")
        cur_optim_path = os.path.join(self.save_path, str(model_id), "optimizer.pt")
        self.model_id_to_per_epoch_accuracy[model_id][self.cur_epoch] = accuracy
        self.model_id_to_per_epoch_model_path[model_id][self.cur_epoch] = cur_model_path
        self.model_id_to_per_epoch_optim_path[model_id][self.cur_epoch] = cur_optim_path
        torch.save(self.main_model.state_dict(), cur_model_path)
        torch.save(self.optim.state_dict(), cur_optim_path)


    def end_of_round_update(self):
        survived = self.eliminate()
        self.breed(survived)


    def get_cur_model_ids(self):
        return [self.arena_id_to_model_id[i] for i in range(self.num_models)]


    def get_cur_model_ages(self):
        cur_model_ids = self.get_cur_model_ids()
        return {index: self.model_id_to_model_status_config[cur_model_id]["age"] if cur_model_id is not None else None for index, cur_model_id in enumerate(cur_model_ids)}


    def get_cur_performance(self):
        cur_model_ids = self.get_cur_model_ids()
        return {index: self.model_id_to_per_epoch_accuracy[cur_model_id][self.cur_epoch] if cur_model_id is not None else None for index, cur_model_id in enumerate(cur_model_ids)}


    def get_cur_model_paths(self):
        cur_model_ids = self.get_cur_model_ids()
        return {index: self.model_id_to_per_epoch_model_path[cur_model_id][self.cur_poch] if cur_model_id is not None else None for index, cur_model_id in enumerate(cur_model_ids)}

    def get_cur_model_status_configs(self):
        cur_model_ids = self.get_cur_model_ids()
        return {index: self.model_id_to_model_status_config[cur_model_id] if cur_model_id is not None else None for index, cur_model_id in enumerate(cur_model_ids)}


    def get_open_arena_ids(self):
        return [arena_id for arena_id in range(self.num_models) if self.arena_id_to_model_id[arena_id] is None]

    def get_open_arena_count(self):
        return len(self.get_open_arena_ids())


    def update_arena_id_to_model_id(self, arena_id, model_id):
        self.arena_id_to_model_id[arena_id] = model_id


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
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.max_weight_deviation = evolution_config["max_weight_deviation"]
        self.breed_intention_rate = evolution_config["breed_intention_rate"]
        self.downward_acceptance = evolution_config["downward_acceptance"]
        self.max_rounds = evolution_config["max_rounds"]

        self.cur_round = evolution_config["start_round"]
        self.cur_epoch = evolution_config["start_epoch"]
        self.cur_arena_id = 0


    def eliminate(self):

        performance = self.get_cur_performance()
        ages = self.get_cur_model_ages()
        performance_list = [value for key, value in performance.items()].sort()

        survival_performance = performance_list[int(len(performance_list) * self.elimination_rate) + 1]
        eliminated = [arena_id for arena_id, accuracy in performance.items() if accuracy < survival_performance and ages[arena_id] < self.max_round_per_model]
        survived = [arena_id for arena_id in performance.keys() if arena_id not in eliminated]
        for arena_id in eliminated:
            self.update_arena_id_to_model_id(arena_id, None)
        return survived

    
    def breed(self, survived):
        performance = self.get_cur_performance()
        # TODO: Save Model Before They Are Trained
        # TODO: Write load model functions 


    


