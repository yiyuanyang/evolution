"""
    Content: Defines the arena in which models compete
    Author: Yiyuan Yang
    Date: April 29th 2020
"""

from yyycode.utils.factory.model_candidate_factory \
    import ModelCandidateFactory
from yyycode.utils.logger.logger import Logger
from yyycode.utils.arena.arena_manager import ArenaManager
from yyycode.utils.lineage.lineage_tree import Lineage
import numpy as np


class Arena(object):
    def __init__(self, data_loaders, train_config, save_config):
        self.am = ArenaManager(data_loaders, train_config, save_config)
        self.mcf = ModelCandidateFactory(train_config, save_config)
        self.model_candidates = {}
        self.logger = Logger(self.am.save_dir())
        for arena_id in range(self.am.num_models()):
            if self.am.evolution_config("use_existing_model"):
                self.model_candidates[arena_id] = \
                    self.mcf.load_model_candidate(arena_id=arena_id)
            else:
                mc, _ = self.mcf.gen_model_candidate(shield=0)
                self.mcf.enter_arena(self, mc, arena_id, new_model=True)

    def get_model_ids(self):
        return [
            model_candidate.mcm.model_id()
            for arena_id, model_candidate in self.model_candidates.items()]

    def get_arena_ids(self):
        return [
            model_candidate.mcm.arena_id()
            for arena_id, model_candidate
            in self.model_candidates.items()]

    def get_model_ages(self):
        return {
            model_candidate.mcm.arena_id(): model_candidate.mcm.age_left()
            for arena_id, model_candidate in self.model_candidates.items()
            if model_candidate is not None}

    def get_shields(self):
        return {
            model_candidate.mcm.arena_id(): model_candidate.mcm.shield() > 0
            for arena_id, model_candidate in self.model_candidates.items()
            if model_candidate is not None}

    def get_accuracy(self, phase=1):
        return {
            model_candidate.mcm.arena_id():
                model_candidate.mcm.cur_accuracy(phase)
            for arena_id, model_candidate in self.model_candidates.items()
            if model_candidate is not None}

    def get_loss(self, phase=1):
        return {
            model_candidate.mcm.arena_id():
                model_candidate.mcm.cur_loss(phase)
            for arena_id, model_candidate in self.model_candidates.items()
            if model_candidate is not None}

    def eliminate(self):
        accuracy_elimination_list = self.am.eliminate_by_accuracy(
            self,
            self.logger)
        age_elimination_list = self.am.eliminate_by_age(self, self.logger)
        elimination_list = list(set(
            accuracy_elimination_list + age_elimination_list))
        return elimination_list

    def breed(self, eliminated):
        top_performers = [
            arena_id
            for arena_id, _ in sorted(
                self.get_accuracy().items(),
                key=lambda x: x[1],
                reverse=True)]
        top_performers = [
            arena_id for arena_id in top_performers
            if arena_id not in eliminated
        ]
        new_candidates = {}
        for i in range(len(top_performers)):
            if i == len(eliminated) or i == len(top_performers) - 1:
                break
            parent_arena_id_1 = top_performers[i]
            parent_arena_id_2 = np.random.choice(
                top_performers[i+1:i+self.am.downward_acceptance()])
            new_candidates[eliminated[i]] = self._breed(
                parent_arena_id_1=parent_arena_id_1,
                parent_arena_id_2=parent_arena_id_2,
                target_arena_id=eliminated[i],
                logger=self.logger)
        replaced = new_candidates.keys()
        if len(replaced) < len(eliminated):
            new_candidates_2 = self.breed(
                [item for item in eliminated if item not in replaced])
            new_candidates.update(new_candidates_2)
        return new_candidates

    def _breed(
        self,
        parent_arena_id_1,
        parent_arena_id_2,
        target_arena_id,
        logger
    ):
        self.logger._set_phase(phase=4)  # ** For Breeding
        new_candidate, new_model_id = self.mcf.gen_model_candidate(
            parent_1_lineage=self.model_candidates[
                parent_arena_id_1].mcm.lineage(),
            parent_2_lineage=self.model_candidates[
                parent_arena_id_2].mcm.lineage())
        new_lineage = Lineage(
            new_model_id,
            self.model_candidates[parent_arena_id_1].mcm.lineage(),
            self.model_candidates[parent_arena_id_2].mcm.lineage())
        self.model_candidates[target_arena_id].breed(
            other_candidate=self.model_candidates[parent_arena_id_2],
            new_arena_id=target_arena_id,
            new_model_id=new_model_id,
            logger=logger,
            policy=self.am.policy(),
            max_weight_mutation=self.am.max_weight_mutation())
        self.logger.log_lineage(target_arena_id, new_model_id, new_lineage)
        self.logger._set_phase(phase=3)  # ** Back to general
        return new_candidate

    def run_round(self):
        for arena_id in self.get_arena_ids():
            self.model_candidates[arena_id].run_round(
                self.am.epoch(),
                self.am.epoch_per_round(),
                self.am.data_loaders)

    def run_experiment(self):
        for i in range(self.am.rounds(), self.am.max_rounds()):
            self.logger.log("Initiating Round {i}".format(i=i))
            self.run_round()
            self.am.epoch_step(self.am.epoch_per_round() - 1)
            self.am.update_stats(self)
            elimination_list = self.eliminate()
            new_candidates = None
            if elimination_list is not None:
                new_candidates = self.breed(elimination_list)
            self.end_of_round_analysis(elimination_list, new_candidates)
            self.eliminate_and_replace(new_candidates)
            self.am.epoch_step(1)

    def eliminate_and_replace(self, new_candidates):
        if new_candidates is None:
            return
        for arena_id, candidate in new_candidates.items():
            candidate.enter_arena(arena_id, self.am.epoch(), new_model=False)
            self.model_candidates[arena_id].mfm.delete_model_optimizer()
            del self.model_candidates[arena_id]
            self.model_candidates[arena_id] = candidate

    def end_of_round_analysis(self, elimination_list, new_candidates=None):
        self.logger.log_end_of_round_state(self, elimination_list, new_candidates)
