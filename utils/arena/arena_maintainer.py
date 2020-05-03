"""
    Content: This class maintains the arena object
    Author: Yiyuan Yang
    Date: April 30th 2020
"""

from Evolution.arena.model_candidate import ModelCandidate, gen_model_candidate_config
from Evolution.utils.lineage.lineage_tree import Lineage
import math
import numpy as np
import pickle
import pandas as pd
import os

class ArenaMaintainer(object):
    def __init__(self, data_loaders, train_config, save_config):
        self.data_loaders = data_loaders
        self.train_config = train_config
        self.save_config = save_config
        self.cur_epoch = self.evolution_config("start_epoch")

    def model_config(self):
        return self.train_config["model_config"]

    def evolution_config(self, key=None):
        if key is None:
            return self.train_config["evolution_config"]
        return self.train_config["evolution_config"][key]

    def backprop_config(self):
        return self.train_config["backprop_config"]

    def create_dir(self, directory):
        if not os.path.exists(directory):
            os.system("mkdir " + directory)
        return directory

    def create_file(self, directory):
        if not os.path.exists(directory):
            os.system("touch " + directory)
        return directory

    def save_dir(self):
        return self.create_dir(self.save_config["save_dir"])

    def model_save_dir(self):
        return self.create_dir(os.path.join(self.save_config["save_dir"], "models"))

    def arena_save_dir(self):
        return self.create_dir(os.path.join(self.save_config["save_dir"], "arena"))

    def original_max_weight_mutation(self):
        return self.evolution_config("max_weight_mutation")

    def max_weight_mutation(self):
        return self.evolution_config("max_weight_mutation") * (self.max_weight_mutation_decay() ** self.epoch())

    def max_weight_mutation_decay(self):
        return self.evolution_config("max_weight_mutation_decay")

    def use_existing_model(self):
        return self.evolution_config("use_existing_model")
        
    def max_round_per_model(self):
        return self.evolution_config("max_round_per_model")

    def age_variation(self):
        return self.evolution_config("age_variation")
    
    def epoch_per_round(self):
        return self.evolution_config("epoch_per_round")

    def num_models(self):
        return self.evolution_config("num_models")

    def load_from_models(self):
        return self.evolution_config("load_from_models")

    def elimination_rate(self):
        return self.evolution_config("elimination_rate")
        
    def mutation_policy(self):
        return self.evolution_config("mutation_policy")
    
    def random_seed(self):
        return self.evolution_config("random_seed")
    
    def shield_epoch(self):
        return self.evolution_config("shield_epoch")

    def downward_acceptance(self):
        return self.evolution_config("downward_acceptance")
    
    def max_rounds(self):
        return self.evolution_config("max_rounds")

    def epoch(self):
        return self.cur_epoch

    def epoch_step(self, steps = None):
        if steps is None:
            steps = self.epoch_per_round()
        self.cur_epoch += steps
    
    def rounds(self):
        return math.ceil(self.epoch() / self.epoch_per_round())
    
    def cur_arena_id(self):
        return self.evolution_config("cur_arena_id")

    """
        Loading the model candidates
    """
    def load_model_candidate(self, arena_id):
        arena_config_save_dir = os.path.join(self.arena_save_dir(), str(arena_id) + "_config.pickle")
        with open(arena_config_save_dir, 'rb') as handle:
            return ModelCandidate(pickle.load(handle))
    
    def init_model_candidate(
        self, 
        model_id, 
        arena_id = None, 
        lineage = None,
        epoch = None, 
        shield = None
    ):
        if lineage is None:
            lineage = Lineage(model_id, None, None)
        if epoch is None:
            epoch = self.epoch() - 1
        if shield is None:
            shield = self.shield_epoch()
        config = gen_model_candidate_config(
            arena_id=arena_id,
            model_id=model_id,
            epoch=epoch,
            save_dir=self.model_save_dir(),
            arena_save_dir=self.arena_save_dir(),
            model_config=self.model_config(),
            backprop_config=self.backprop_config(),
            random_seed=self.random_seed() + model_id,
            age_left = self.max_round_per_model() + np.random.choice(self.age_variation()),
            shield = shield,
            lineage = lineage
        )
        mc = ModelCandidate(config)
        return mc

    """
        Eliminating Models
    """
    def eliminate_by_age(self, arena, logger):
        ages = arena.get_model_ages()
        shielded = arena.get_shielded_ids()
        survive_list = []
        eliminate_list = []
        for arena_id, age_left in ages.items():
            assert age_left >=0, "Age Left Should Not Got Negative"
            if age_left == 0 and not shielded[arena_id]:
                eliminate_list.append(arena_id)
            else:
                survive_list.append(arena_id)
        logger.log_elimination(survive_list, eliminate_list, ages, "Age")
        return eliminate_list

    def eliminate_by_accuracy(self, arena, logger):
        accuracies = arena.get_eval_accuracies()
        raw_accuracies = sorted([value for key, value in accuracies.items()])
        cut_off = raw_accuracies[int(round(len(raw_accuracies)*self.elimination_rate()))]
        shielded = arena.get_shielded_ids()
        survive_list = []
        eliminate_list = []
        for arena_id, accuracy in accuracies.items():
            if accuracy < cut_off and not shielded[arena_id]:
                eliminate_list.append(arena_id)
            else:
                survive_list.append(arena_id)
        logger.log_elimination(survive_list, eliminate_list, accuracies, "Performance")
        return eliminate_list

    
    def update_stats(self, arena):
        train_save_dir = self.create_file(os.path.join(self.arena_save_dir(), "train_stats.csv"))
        eval_save_dir = self.create_file(os.path.join(self.arena_save_dir(), "eval_stats.csv"))
        test_save_dir = self.create_file(os.path.join(self.arena_save_dir(), "test_stats.csv"))
        if self.use_existing_model():
            train_stats_dict = self.load_dataframe(pd.read_csv(train_save_dir))
            eval_stats_dict = self.load_dataframe(pd.read_csv(eval_save_dir))
            test_stats_dict = self.load_dataframe(pd.read_csv(test_save_dir))
        else:
            train_stats_dict = self.init_stats_dict()
            eval_stats_dict = self.init_stats_dict()
            test_stats_dict = self.init_stats_dict()
        train_stats_dict = self._update_stats(stats_dict=train_stats_dict, arena=arena, phase=0)
        eval_stats_dict = self._update_stats(stats_dict=eval_stats_dict, arena=arena, phase=1)
        test_stats_dict = self._update_stats(stats_dict=test_stats_dict, arena=arena, phase=2)
        train_stats_dict.to_csv(train_save_dir, index = False)
        eval_stats_dict.to_csv(eval_save_dir, index = False)
        test_stats_dict.to_csv(test_save_dir, index = False)


    def _update_stats(self, stats_dict, arena, phase):
        accuracies = arena.get_accuracies(phase)
        losses = arena.get_losses(phase)
        if self.epoch() not in stats_dict["epoch"]:
            stats_dict["epoch"].append(self.epoch())
            for arena_id, accuracy in accuracies.items():
                stats_dict["id: " + str(int(arena_id)) + " accuracy"].append(accuracy)
                stats_dict["id: " + str(int(arena_id)) + " loss"].append(losses[arena_id])
        return pd.DataFrame.from_dict(stats_dict)


    def load_dataframe(self, df):
        stats_dict = dict()
        stats_dict["epoch"] = df["epoch"].tolist()
        for arena_id in range(self.num_models()):
            col_name = "id: " + str(int(arena_id))
            stats_dict[col_name + " accuracy"] = df[col_name + " accuracy"].tolist()
            stats_dict[col_name + " loss"] = df[col_name + " loss"].tolist()
        return stats_dict

    def init_stats_dict(self):
        stats_dict = dict()
        stats_dict["epoch"] = []
        for arena_id in range(self.num_models()):
            stats_dict["id: " + str(int(arena_id)) + " accuracy"] = []
            stats_dict["id: " + str(int(arena_id)) + " loss"] = []
        return stats_dict











    