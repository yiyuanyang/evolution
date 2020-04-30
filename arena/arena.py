"""
    Content: Defines the arena in which models compete
    Author: Yiyuan Yang
    Date: April 29th 2020
"""

import os
import torch
import Evolution.model.models.resnet.arena_resnet as arena_resnet
from Evolution.model.models.resnet.arena_resnet import gen_model
import numpy as np


class Arena(object):
    def __init__(self, train_config, save_config):
        self.init_record()
        evolution_config = train_config["evolution_config"]
        self.init_evolution_record(evolution_config)
        self.init_main_model(train_config)
        self.init_secondary_model(train_config)
        # Track Each Model
        self.num_models = evolution_config["num_models"]
        for i in range(self.num_models):
            self.arena_id_to_model_id[i] = i
            cur_path = os.system("mkdir " + os.path.join(save_config["model_save_dir"], i))
            self.model_id_to_per_epoch_model_path[i] = cur_path
            self.model_id_to_model_status_config[i] = arena_resnet.gen_model_status_config(
                initial_model=True,
                model_id=i,
                arena_id=i,
                lineage=[None, None],
                age=0
            )
        self.new_model_id = self.num_models


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


    def get_cur_model_ids(self):
        return [self.arena_id_to_model_id[i] for i in range(self.num_models)]


    def get_cur_epoch_performance(self):
        cur_model_ids = self.get_cur_model_ids()
        return {index: self.model_id_to_per_epoch_accuracy[cur_model_id][self.epoch] if cur_model_id is not None else None for index, cur_model_id in enumerate(cur_model_ids)}


    def get_cur_model_paths(self):
        cur_model_ids = self.get_cur_model_ids()
        return {index: self.model_id_to_per_epoch_model_path[cur_model_id][self.epoch] if cur_model_id is not None else None for index, cur_model_id in enumerate(cur_model_ids)}

    def get_cur_model_status_configs(self):
        cur_model_ids = self.get_cur_model_ids()
        return {index: self.model_id_to_model_status_config[cur_model_id] if cur_model_id is not None else None for index, cur_model_id in enumerate(cur_model_ids)}


    def get_open_arena_ids(self):
        return [arena_id for arena_id in range(self.num_models) if self.arena_id_to_model_id[arena_id] is None]

    def get_open_arena_count(self):
        return len(self.get_open_arena_ids())


    def update_arena_id_to_model_id(self, arena_id, model_id):
        self.arena_id_to_model_id[arena_id] = model_id


    def init_evolution_record(self, evolution_config):
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
        self.round = evolution_config["start_round"]
        self.epoch = evolution_config["start_epoch"]

    def eliminate(self):
        performance = self.get_cur_epoch_performance()
        performance_list = [value for key, value in performance.items()].sort()
        survival_performance = performance_list[int(len(performance_list) * self.elimination_rate) + 1]
        eliminated = [arena_id for arena_id, accuracy in performance.items() if accuracy < survival_performance]
        survived = [arena_id for arena_id in performance.keys() if arena_id not in eliminated]
        for arena_id in eliminated:
            self.update_arena_id_to_model_id(arena_id, None)
        return survived

    


