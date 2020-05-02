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
from Evolution.arena.model_candidate import ModelCandidate, gen_model_candidate_config
from Evolution.utils.log.logger import Logger
from Evolution.arena.arena_maintainer import ArenaMaintainer
from sklearn import metrics
from Evolution.utils.lineage.lineage_tree import Lineage
import numpy as np
import pandas as pd
import pickle
import copy

class Arena(object):
    def __init__(self, data_loaders, train_config, save_config):
        self.am = ArenaMaintainer(data_loaders, train_config, save_config)
        self.model_candidates = {}
        self.logger = Logger(self.am.save_dir())
        self.new_model_id = self.am.evolution_config("cur_arena_id")
        np.random.seed(self.am.random_seed()-1)
        for arena_id in range(self.am.num_models()):
            if self.am.evolution_config("use_existing_model"):
                self.model_candidates[arena_id] = self.am.load_model_candidate(arena_id)
            else:
                self.model_candidates[arena_id] = self.am.init_model_candidate(arena_id, self.gen_new_model_id(), shield = 0)

    def gen_new_model_id(self):
        model_id = self.new_model_id
        self.new_model_id += 1
        return model_id

    def get_model_ids(self):
        return [model_candidate.mm.model_id() for arena_id, model_candidate in self.model_candidates.items()]

    def get_arena_ids(self):
        return [model_candidate.mm.arena_id() for arena_id, model_candidate in self.model_candidates.items() if model_candidate is not None]

    def get_model_ages(self):
        return {model_candidate.mm.arena_id(): model_candidate.mm.age_left() for arena_id, model_candidate in self.model_candidates.items() if model_candidate is not None}

    def get_shielded_ids(self):
        return {model_candidate.mm.arena_id(): model_candidate.mm.shield()>0 for arena_id, model_candidate in self.model_candidates.items() if model_candidate is not None}

    def get_train_accuracies(self):
        return {model_candidate.mm.arena_id(): model_candidate.mm.accuracy(0, self.am.epoch()) for arena_id, model_candidate in self.model_candidates.items() if model_candidate is not None}

    def get_eval_accuracies(self):
        return {model_candidate.mm.arena_id(): model_candidate.mm.accuracy(1, self.am.epoch()) for arena_id, model_candidate in self.model_candidates.items() if model_candidate is not None}

    def get_test_accuracies(self):
        return {model_candidate.mm.arena_id(): model_candidate.mm.accuracy(0, self.am.epoch()) for arena_id, model_candidate in self.model_candidates.items() if model_candidate is not None}

    def get_train_losses(self):
        return {model_candidate.mm.arena_id(): model_candidate.mm.loss(0, self.am.epoch()) for arena_id, model_candidate in self.model_candidates.items() if model_candidate is not None}

    def get_eval_losses(self):
        return {model_candidate.mm.arena_id(): model_candidate.mm.loss(1, self.am.epoch()) for arena_id, model_candidate in self.model_candidates.items() if model_candidate is not None}

    def get_test_losses(self):
        return {model_candidate.mm.arena_id(): model_candidate.mm.loss(2, self.am.epoch()) for arena_id, model_candidate in self.model_candidates.items() if model_candidate is not None}

    def get_open_arena_ids(self):
        return [arena_id for arena_id in range(self.am.num_models()) if self.model_candidates[arena_id] is None]

    def eliminate(self):
        self.am.eliminate_by_accuracy(self, self.logger)
        self.am.eliminate_by_age(self, self.logger)

    def breed(self):
        survived = self.get_arena_ids()
        eliminated = self.get_open_arena_ids()
        for i in range(len(survived)):
            if i == len(eliminated) or i == len(survived) - 1:
                break
            parent_arena_id_1 = survived[i]
            parent_arena_id_2 = np.random.choice(survived[i+1:i+6])
            self._breed(parent_arena_id_1, parent_arena_id_2, eliminated[i], self.logger)
        if len(survived) <= len(eliminated):
            self.logger.log("Another Round Breeding")
            self.breed()

    def _breed(
        self, 
        parent_arena_id_1, 
        parent_arena_id_2, 
        target_arena_id,
        logger
    ):
        new_model_id = self.gen_new_model_id()
        new_lineage = Lineage(
            new_model_id, 
            self.model_candidates[parent_arena_id_1].mm.lineage(),
            self.model_candidates[parent_arena_id_2].mm.lineage()
        )
        self.model_candidates[target_arena_id] = self.am.init_model_candidate(target_arena_id, new_model_id, new_lineage)
        self.model_candidates[target_arena_id].breed(
            other_candidate = self.model_candidates[parent_arena_id_2], 
            new_arena_id = target_arena_id,
            new_model_id = new_model_id, 
            logger = logger,
            mutation_policy = self.am.mutation_policy(),
            max_weight_mutation = self.am.max_weight_mutation()
        )

    def run_round(self):
        for arena_id in self.get_arena_ids():
            self.model_candidates[arena_id].run_round(self.am.epoch(), self.am.epoch_per_round(), self.am.data_loaders)


    
    def run_experiment(self):
        for i in range(self.am.rounds(), self.am.max_rounds()):
            self.logger.log("Initiating Round {i}".format(i=i))
            self.run_round()
            self.am.epoch_step(self.am.epoch_per_round() - 1)
            self.am.update_stats(self)
            np.random.seed(self.am.random_seed()-1)
            self.eliminate()
            self.breed()
            self.am.epoch_step(1)

    

        

        


    


