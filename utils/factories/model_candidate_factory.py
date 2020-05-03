"""
    Content: This class generates model candidates
    Author: Yiyuan Yang
    Date: May 2nd 2020
"""

from Evolution.arena.model_maintainer import ModelMaintainer
from Evolution.arena.model_candidate import ModelCandidate
from Evolution.utils.lineage.lineage_tree import Lineage
import numpy as np
import pickle
import os

class ModelCandidateFactory(object):
    def __init__(
        self,
        train_config,
        arena_save_dir,
        all_model_save_dir,
        next_model_id,
    ):
        self.train_config = self._create_dir_if_not_exist(train_config)
        self.arena_save_dir = self._create_dir_if_not_exist(arena_save_dir)
        self.all_model_save_dir = self._create_dir_if_not_exist(all_model_save_dir)
        self.next_model_id = next_model_id

    def _new_model_id(self):
        model_id = self.next_model_id
        self.next_model_id += 1
        return model_id
    
    def _create_dir_if_not_exist(self, directory):
        if not os.path.exists(directory):
            os.system("mkdir " + directory)
        return directory


    def load_model_candidate(self, arena_id):
        model_candidate_config_path = os.path.join(self.arena_save_dir, str(arena_id) + "_config.pickle")
        assert os.path.exists(model_candidate_config_path), "Candidate Does Not Exist In Arena"
        with open(model_candidate_config_path, 'rb') as handle:
            config = pickle.load(handle)
        new_candidate = ModelCandidate(config)
        return new_candidate


    def gen_model_candidate(self):
        # ** Basic Settings Other Than Arena ID and Epoch, Will Assign These When We Place the Candidate Into Arena
        model_id = self._new_model_id()
        evolution_config = self.train_config["evolution_config"]
        random_seed = evolution_config["random_seed"]
        np.random.seed(random_seed)
        config = {
            "model_id": model_id,
            "model_save_dir": os.path.join(self.all_model_save_dir, str(model_id)),
            "arena_save_dir": self.arena_save_dir,
            "model_config": self.train_config["model_config"],
            "backprop_config": self.train_config["evolution_config"],
            "random_seed": evolution_config["random_seed"],
            "age_left": evolution_config["max_round_per_model"] + \
                np.random.choice(list(range(-evolution_config["age_variation"], evolution_config["age_variation"]))),
            "shield_epoch": evolution_config["shield_epoch"] + \
                np.random.choice(list(range(-evolution_config["age_variation"], evolution_config["shield_variation"]))),
            "lineage": Lineage(model_id, None, None)
        }
        return ModelCandidate(config)

        

