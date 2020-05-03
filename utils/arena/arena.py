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
from Evolution.utils.factories.model_candidate_factory import ModelCandidateFactory
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
        self.mcf = ModelCandidateFactory(train_config, save_config)
        self.model_candidates = {}
        self.logger = Logger(self.am.save_dir())
        np.random.seed(self.am.random_seed()-1)
        for arena_id in range(self.am.num_models()):
            if self.am.evolution_config("use_existing_model"):
                self.model_candidates[arena_id] = self.mcf.load_model_candidate(arena_id=arena_id)
            else:
                mc = self.mcf.gen_model_candidate(shield=0)
                self.mcf.enter_arena(self, mc, arena_id)

    def get_model_ids(self):
        return [model_candidate.mm.model_id() for arena_id, model_candidate in self.model_candidates.items()]

    def get_arena_ids(self):
        return [model_candidate.mm.arena_id() for arena_id, model_candidate in self.model_candidates.items()]

    def get_model_ages(self):
        return {model_candidate.mm.arena_id(): model_candidate.mm.age_left() for arena_id, model_candidate in self.model_candidates.items() if model_candidate is not None}

    def get_shielded_ids(self):
        return {model_candidate.mm.arena_id(): model_candidate.mm.shield()>0 for arena_id, model_candidate in self.model_candidates.items() if model_candidate is not None}

    def get_accuracies(self, phase = 0):
        return {model_candidate.mm.arena_id(): model_candidate.mm.accuracy(phase, self.am.epoch()) for arena_id, model_candidate in self.model_candidates.items() if model_candidate is not None}

    def get_losses(self, phase = 0):
        return {model_candidate.mm.arena_id(): model_candidate.mm.loss(phase, self.am.epoch()) for arena_id, model_candidate in self.model_candidates.items() if model_candidate is not None}

    def eliminate(self):
        accuracy_elimination_list = self.am.eliminate_by_accuracy(self, self.logger)
        age_elimination_list = self.am.eliminate_by_age(self, self.logger)
        elimination_list = list(set(accuracy_elimination_list + age_elimination_list))
        return elimination_list

    def breed(self, eliminated):
        survived = [arena_id for arena_id in self.get_arena_ids() if arena_id not in eliminated]
        new_candidates = {}
        for i in range(len(survived)):
            if i == len(eliminated) or i == len(survived) - 1:
                break
            parent_arena_id_1 = survived[i]
            parent_arena_id_2 = np.random.choice(survived[i+1:i+6])
            new_candidates[eliminated[i]] = self._breed(
                parent_arena_id_1=parent_arena_id_1, 
                parent_arena_id_2=parent_arena_id_2, 
                target_arena_id=eliminated[i], 
                logger=self.logger
            )
        replaced = new_candidates.keys()
        if len(replaced) < len(eliminated):
            # ** Breed Again If Not All Slots Are Filled
            new_candidates_2 = self.breed(
                [item for item in eliminated if item not in replaced]
            )
            new_candidates.update(new_candidates_2)
        return new_candidates

    def _breed(
        self, 
        parent_arena_id_1, 
        parent_arena_id_2, 
        target_arena_id,
        logger
    ):
        # ** Generate the New Candidate
        new_candidate, new_model_id = self.mcf.gen_model_candidate(
            parent_1_lineage=self.model_candidates[parent_arena_id_1].mcm.lineage(),
            parent_2_lineage=self.model_candidates[parent_arena_id_2].mcm.lineage())
        new_lineage = Lineage(
            new_model_id, 
            self.model_candidates[parent_arena_id_1].mcm.lineage(),
            self.model_candidates[parent_arena_id_2].mcm.lineage()
        )
        # ** Store the Model
        self.model_candidates[target_arena_id].breed(
            other_candidate = self.model_candidates[parent_arena_id_2], 
            new_arena_id = target_arena_id,
            new_model_id = new_model_id, 
            logger = logger,
            mutation_policy = self.am.mutation_policy(),
            max_weight_mutation = self.am.max_weight_mutation()
        ) 
        self.logger.log_lineage(target_arena_id, new_lineage)
        return new_candidate

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
            elimination_list = self.eliminate()
            new_candidates = self.breed(elimination_list)
            self.eliminate_and_replace(new_candidates)
            self.am.epoch_step(1)

    def eliminate_and_replace(self, new_candidates):
        for arena_id, candidate in new_candidates:
            del self.model_candidates[arena_id]
            self.model_candidates[arena_id] = candidate

    

        

        


    


