"""
    Content: This class generates model candidates
    Author: Yiyuan Yang
    Date: May 2nd 2020
"""

from Evolution.arena.model_candidate import ModelCandidate
from Evolution.utils.lineage.lineage_tree import Lineage
import numpy as np
import pickle
import os

class ModelCandidateFactory(object):
    def __init__(self, train_config, save_config):
        self.train_config = train_config
        save_dir = save_config["save_dir"]
        self.arena_save_dir = self._create_dir_if_not_exist(os.path.join(save_dir, "arena"))
        self.all_model_save_dir = self._create_dir_if_not_exist(os.path.join(save_dir, "models"))
        self.next_model_id = self.train_config["evolution_config"]["cur_model_id"]

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

    def save_model_candidate(self, arena_id, config):
        model_candidate_config_path = os.path.join(self.arena_save_dir, str(arena_id) + "_config.pickle")
        with open(model_candidate_config_path, 'wb') as handle:
            pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def gen_model_candidate(self, shield = None, parent_1_lineage = None, parent_2_lineage = None):
        # ** Basic Settings Other Than Arena ID and Epoch, Will Assign These When We Place the Candidate Into Arena
        model_id = self._new_model_id()
        evolution_config = self.train_config["evolution_config"]
        random_seed = evolution_config["random_seed"]
        np.random.seed(random_seed)
        if shield is None:
            shield = evolution_config["shield_epoch"] + \
                np.random.choice(list(range(-evolution_config["age_variation"], evolution_config["shield_variation"])))
        if parent_1_lineage is None:
            lineage = Lineage(model_id, None, None)
        else:
            lineage = Lineage(model_id, parent_1_lineage, parent_2_lineage)
        config = {
            "model_id": model_id,
            "model_save_dir": os.path.join(self.all_model_save_dir, str(model_id)),
            "arena_save_dir": self.arena_save_dir,
            "model_config": self.train_config["model_config"],
            "backprop_config": self.train_config["evolution_config"],
            "random_seed": evolution_config["random_seed"],
            "age_left": evolution_config["max_round_per_model"] + \
                np.random.choice(list(range(-evolution_config["age_variation"], evolution_config["age_variation"]))),
            "shield_epoch": shield,
            "lineage": lineage
        }
        return ModelCandidate(config), model_id

    
    def enter_arena(self, arena, model_candidate, arena_id):
        epoch = arena.am.epoch() - 1
        config = model_candidate.enter_arena(arena_id, epoch)
        if arena_id in arena.model_candidates.keys() and arena.model_candidates[arena_id] is not None:
            del arena.model_candidates[arena_id]
        arena.model_candidates[arena_id] = model_candidate
        self.save_model_candidate(arena_id, config)




        

