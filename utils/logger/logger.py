"""
    Content: Logger object that can be passed around to do logging
    Author: Yiyuan Yang
    Date: April. 18th 2020
"""


from Evolution.utils.weights import weight_stats
from Evolution.model.model_components.resnet_components.residual_block \
    import BasicBlock, Bottleneck
from scipy import special as sp
from sklearn import metrics
import pandas as pd
import numpy as np
import yaml
import json
import os


class Logger(object):
    def __init__(self,
                 logger_save_dir,
                 dump_frequency=1,
                 print_to_console=True):
        self.messages = []
        self.dump_frequency = dump_frequency
        self.phase = 3
        self.print_to_console = print_to_console

        assert os.path.exists(logger_save_dir), \
            "FATAL: Save Directory Does not Exist"

        self.logger_save_dir = [
            os.path.join(logger_save_dir, "train.log"),
            os.path.join(logger_save_dir, "eval.log"),
            os.path.join(logger_save_dir, "test.log"),
            os.path.join(logger_save_dir, "general.log"),
            os.path.join(logger_save_dir, "breed.log")
        ]
        self.stat_save_dir = [
            os.path.join(logger_save_dir, "train.csv"),
            os.path.join(logger_save_dir, "eval.csv"),
            os.path.join(logger_save_dir, "test.csv")
        ]
        self.stats = [
            self._gen_stats_container(),
            self._gen_stats_container(),
            self._gen_stats_container()]

    def _gen_stats_container(self):
        """return a container for stats

        Returns:
            {dict} -- container for stats
        """
        return {
            "epoch": [],
            "global_accuracy": [],
            "accuracy": [],
            "global_recall": [],
            "recall": [],
            "global_precision": [],
            "precision": [],
            "global_f1": [],
            "f1": [],
            "loss": []
        }

    # Basic helper functions
    def _log(self, log, force_dump=False):
        """Add Message To Queue, Dump If Necessary

        Arguments:
            log {string} -- message to be logged

        Keyword Arguments:
            force_dump {bool} -- whether to force print to console
                (default: {False})
        """
        self.messages.append("\n" + log)
        if force_dump or len(self.messages) >= self.dump_frequency:
            self._dump()
        if self.print_to_console:
            print(log)

    def set_phase(self, epoch, phase):
        """Logs the phase switching during train/eval/test

        Arguments:
            epoch {int} -- Current Epoch
            phase {int} -- Phase: Train: 0, Eval: 1, Test: 2
                Nothing prints to console during test
        """

        if len(self.messages) != 0:
            self._dump()
        if phase != 2:
            self.print_to_console = True
        phases = ["Train", "Eval", "Test"]
        self.phase = phase
        self.log(
            "======================SWITCHING PHASE=========================")
        log = "LOGGING: Starting " + phases[phase] + " phase for epoch " + str(
            epoch)
        self._log(log=log)
        if phase == 2:
            self.print_to_console = False

    def _set_phase(self, phase):
        self.phase = phase

    def _dump(self):
        """Print Messages to console
        """

        file = open(self.logger_save_dir[self.phase], "a")
        file.writelines(self.messages)
        self.messages = []

    def log(self, message):
        """Basic Logging

        Arguments:
            message {string} -- log message
        """

        self._log(log="LOGGING: " + message)

    def log_learning_rate_change(self, epoch, cur, new):
        """Logs changes in learning rate

        Arguments:
            epoch {int} -- Current Epoch
            cur {float} -- Current Learning Rate
            new {float} -- New Learning Rate
        """

        log = ("LOGGING: Epoch " + str(epoch) +
               " , adjusted learning rate from " + str(cur) + " to " +
               str(new))
        self.log("===LEARNING RATE===")
        self._log(log=log)

    def log_config(self, config_name, config):
        """Logs A Config

        Arguments:
            config_name {string} -- which is this config for?
            config {dict()} -- config dictionary
        """

        log = "LOGGING: " + config_name + ": " + yaml.dump(config)
        self.log("===MODEL CONFIG===")
        self._log(log=log)

    def log_batch_result(self,
                         batch_index,
                         num_batches,
                         prediction_prob,
                         prediction,
                         ground_truth,
                         loss,
                         top_n=5):
        """Logs Result For Every Batch

        Arguments:
            batch_index {int} -- Index of current batch
            num_batches {int} -- Total number of batches per epoch
            prediction_prob {list[float]} -- Predicted Probability
                for each class
            prediction {list[float]} -- Assigning max to 1 and rest
                to 0
            ground_truth {list[long]} -- ground truth
            loss {list[float]} -- calculated loss between
                prediction_prob and ground_truth

        Keyword Arguments:
            top_n {int} -- if batch is too large, only log top_n
                (default: {5})
        """

        prediction_prob = [
            self.round_to_4_decimal(item) for item in prediction_prob
        ]
        prediction_softmax_string = \
            [str(self.round_to_4_decimal(sp.softmax(item))) for item in prediction_prob]
        prediction_softmax_string = "\n".join(
            prediction_softmax_string[:top_n])
        prediction_string = [str(item) for item in prediction_prob]
        prediction_string = "\n".join(prediction_string[:top_n])
        log = "LOGGING: Batch " + str(batch_index) + "/" + str(num_batches) + \
            "\nFor First {top_n}".format(top_n=top_n) + \
            "\n Prediction Prob: \n" + prediction_string + \
            "\n Prediction Softmax: \n" + prediction_softmax_string + \
            "\n Prediction:" + str(prediction[:top_n]) + \
            "\n Ground Truth: " + str(ground_truth[:top_n]) + \
            "\n Batch Loss: " + str(loss)

        self._log(log=log)

    def log_data(self, batch_index, data, label):
        """Logs Shape and Value Of the Data

        Arguments:
            batch_index {int} -- Current Batch Index
            data {ndarray} -- Data inputed into model
            label {ndarray} -- ground truth
        """

        log = ("LOGGING: Batch " + str(batch_index) + " Data Shape: " +
               str(list(data.shape)) + " Label Shape: " +
               str(list(label.shape)))
        self.log("===DATA SHAPE===")
        self._log(log=log)

    def _log_accuracy(self, epoch, ground_truth, prediction):
        """
            Logs the confusion matrix
        """
        global_accuracy = metrics.accuracy_score(ground_truth, prediction)

        accuracy_scores = []
        for i in range(0, max(ground_truth) + 1):
            label = [element for element in ground_truth if element == i]
            pred = [
                prediction[j] for j in range(len(ground_truth))
                if ground_truth[j] == i
            ]
            accuracy_scores.append(metrics.accuracy_score(label, pred))
        accuracy_scores = self.round_to_2_decimal(accuracy_scores)

        log = ("LOGGING: Epoch " + str(epoch) + " Global Accuracy : " +
               str(global_accuracy) + "\nAccuracy Scores : " +
               str(accuracy_scores))

        self._log(log=log)

        return global_accuracy, accuracy_scores

    def _log_f1(self, epoch, ground_truth, prediction):
        """
            Logs the confusion matrix
        """

        f1_scores = metrics.f1_score(ground_truth, prediction, average=None)
        global_f1 = round(np.mean(f1_scores), 2)
        f1_scores = self.round_to_2_decimal(f1_scores)

        log = ("LOGGING: Epoch " + str(epoch) + " Global F1: " +
               str(global_f1) + "\nF1 Scores: " + str(f1_scores))

        self._log(log=log)

        return global_f1, f1_scores

    def _log_recall(self, epoch, ground_truth, prediction):
        """
            Logs the confusion matrix
        """
        recall_scores = metrics.recall_score(ground_truth,
                                             prediction,
                                             average=None)
        global_recall = round(np.mean(recall_scores), 2)
        recall_scores = self.round_to_2_decimal(recall_scores)
        log = ("LOGGING: Epoch " + str(epoch) + " Globall Recall: " +
               str(global_recall) + "\nRecall Scores: " + str(recall_scores))

        self._log(log=log)

        return global_recall, recall_scores

    def _log_precision(self, epoch, ground_truth, prediction):
        """
            Logs the confusion matrix
        """
        precision_scores = metrics.precision_score(ground_truth,
                                                   prediction,
                                                   average=None)
        global_precision = np.mean(precision_scores)
        precision_scores = self.round_to_2_decimal(precision_scores)
        log = ("LOGGING: Epoch " + str(epoch) + " Global Precision: " +
               str(global_precision) + "\nPrecision Scores: " +
               str(precision_scores))

        self._log(log=log)

        return global_precision, precision_scores

    def _log_loss(self, epoch, loss):
        loss = np.mean(loss)
        log = ("LOGGING: Epoch " + str(epoch) + " Loss: " + str(loss))
        self._log(log=log)
        return loss

    def _log_confusion_matrix(self, epoch, ground_truth, prediction):
        """
            Logs the confusion matrix
        """
        confusion_matrix = metrics.confusion_matrix(ground_truth, prediction)
        log = ("LOGGING: Epoch " + str(epoch) + "\n Confusion Matrix: \n" +
               str(confusion_matrix))

        self._log(log=log)

    def _log_epoch_metrics(self, epoch, ground_truth, prediction, loss):
        ground_truth = [int(item) for item in ground_truth]
        prediction = [int(item) for item in prediction]

        self.log("===================EPOCH SUMMARY=====================")

        global_accuracy, accuracy = self._log_accuracy(epoch, ground_truth,
                                                       prediction)
        global_recall, recall = self._log_recall(epoch, ground_truth,
                                                 prediction)
        global_precision, precision = self._log_precision(
            epoch, ground_truth, prediction)
        global_f1, f1 = self._log_f1(epoch, ground_truth, prediction)
        global_loss = self._log_loss(epoch, loss)
        self._log_confusion_matrix(epoch, ground_truth, prediction)
        return global_accuracy, accuracy, global_recall, recall, \
            global_precision, precision, global_f1, f1, global_loss

    def log_epoch_metrics(self, epoch, ground_truth, prediction, loss):
        """
            Perform all metrics
        """
        global_accuracy, accuracy, global_recall, recall, \
            global_precision, precision, global_f1, f1, global_loss = \
            self._log_epoch_metrics(epoch, ground_truth, prediction, loss)
        if os.path.exists(self.stat_save_dir[self.phase]):
            self.load_prior_metrics()

        self.stats[self.phase]["epoch"].append(epoch)
        self.stats[self.phase]["global_accuracy"].append(
            round(global_accuracy, 2))
        self.stats[self.phase]["accuracy"].append(accuracy)
        self.stats[self.phase]["global_recall"].append(global_recall)
        self.stats[self.phase]["recall"].append(recall)
        self.stats[self.phase]["global_precision"].append(global_precision)
        self.stats[self.phase]["precision"].append(precision)
        self.stats[self.phase]["global_f1"].append(global_f1)
        self.stats[self.phase]["f1"].append(f1)
        self.stats[self.phase]["loss"].append(global_loss)

        stats_dataframe = pd.DataFrame.from_dict(self.stats[self.phase])
        stats_dataframe.to_csv(self.stat_save_dir[self.phase])
        return global_accuracy, global_loss

    def load_prior_metrics(self):
        train_stats = pd.read_csv(self.stat_save_dir[0])
        eval_stats = pd.read_csv(self.stat_save_dir[1])
        test_stats = pd.read_csv(self.stat_save_dir[2])
        prior_metrics = [train_stats, eval_stats, test_stats]
        for i in range(3):
            self.stats[i]["epoch"] = prior_metrics[i]["epoch"].tolist()
            self.stats[i]["global_accuracy"] = prior_metrics[i][
                "global_accuracy"].tolist()
            self.stats[i]["accuracy"] = self._json_parse(
                prior_metrics[i]["accuracy"].tolist())
            self.stats[i]["global_recall"] = prior_metrics[i][
                "global_recall"].tolist()
            self.stats[i]["recall"] = self._json_parse(
                prior_metrics[i]["recall"].tolist())
            self.stats[i]["global_precision"] = prior_metrics[i][
                "global_precision"].tolist()
            self.stats[i]["precision"] = self._json_parse(
                prior_metrics[i]["precision"].tolist())
            self.stats[i]["global_f1"] = prior_metrics[i]["global_f1"].tolist()
            self.stats[i]["f1"] = self._json_parse(
                prior_metrics[i]["f1"].tolist())
            self.stats[i]["loss"] = prior_metrics[i]["loss"].tolist()

    def _json_parse(self, metric_list):
        return [json.loads(item) for item in metric_list]

    # Model Related Logging
    def log_model(self, model):
        model_log = str(model)
        self.log("===================MODEL===================")
        self.log("Current Model Structure:")
        self.log(model_log)

    def log_model_statistics(self, model, model_name, calculate_statistics):
        self._log("===WEIGHT STATS===")
        calculate_statistics(model, model_name, self)

    def log_residual_block_statistics(self, block):
        if isinstance(block, BasicBlock):
            self._log("Basicblock: ")
            weight_statistics, grad_statistics = \
                weight_stats.basic_block_statistics(block)
        elif isinstance(block, Bottleneck):
            self._log("Bottleneck: ")

        self.log_tensor_statistics(weight_statistics, "weight")
        if grad_statistics is not None:
            self.log_tensor_statistics(grad_statistics, "grad  ")

    def log_tensor_statistics(self, statistics, name):
        max_val, min_val, range_val, mean_val, stdev = statistics
        msg = "{name}".format(name=name)
        msg += " max {max_val} min {min_val} range {range_val}".format(
            max_val=np.format_float_scientific(
                max_val.data.cpu().numpy(),
                precision=2),
            min_val=np.format_float_scientific(
                min_val.data.cpu().numpy(),
                precision=2),
            range_val=np.format_float_scientific(
                range_val.data.cpu().numpy(),
                precision=2)) + \
            " mean {mean_val} stdev {stdev}".format(
            mean_val=np.format_float_scientific(
                mean_val.data.cpu().numpy(),
                precision=2),
            stdev=np.format_float_scientific(
                stdev.data.cpu().numpy(),
                precision=2))
        self._log(msg)

    def log_breed(self, model_id_one, model_id_two, new_model_id):
        self._log("===BREEDING===")
        self._log(
            "Model {model_id_one} and {model_id_two} generated {new_model_id}".
            format(model_id_one=model_id_one,
                   model_id_two=model_id_two,
                   new_model_id=new_model_id))

    def round_to_4_decimal(self, stats):
        return [round(item, 4) for item in stats]

    def round_to_2_decimal(self, stats):
        return [round(item, 2) for item in stats]

    def log_elimination(self, survived, eliminated, value_dict, reason):
        survived = {arena_id: value_dict[arena_id] for arena_id in survived}
        eliminated = {
            arena_id: value_dict[arena_id]
            for arena_id in eliminated
        }
        self.log("===ELIMINATION===")
        self._log("Eliminated by {reason}".format(reason=reason))
        self._log("Survived Value Pairs {survived}".format(survived=survived))
        self._log("Eliminated Value Pairs {eliminated}".format(
            eliminated=eliminated))

    def log_round_stats(self, round, accuracy, loss):
        self.log("===ROUND STATS===")
        self._log("Accuracies For Each Arena_ID: {accuracy}".format(accuracy))
        self._log("Losses For Each Arena_ID: {loss}".format(loss))

    def log_model_activity(self, activity, model_candidate):
        self.log("===MODEL ACTIVITY===")
        self._log(
            "{activity} for Arena ID: {arena_id}, Model ID: {model_id}".format(
                activity=activity,
                arena_id=model_candidate.mcm.arena_id(),
                model_id=model_candidate.mcm.model_id()))

    def log_lineage(self, arena_id, model_id, lineage):
        self.log("===LINEAGE===")
        self._log("Arena ID: {arena_id} Model ID: {model_id} lineage: {lineage}".format(
            arena_id=arena_id, model_id=model_id, lineage=lineage))

    def log_end_of_round_state(self, arena, eliminated, new_candidates=None):
        self._log("==========================================================")
        self._log("==================END OF ROUND ANALYSIS===================")
        self._log("==========================================================")
        cur_round = arena.am.rounds()
        self._log("Round: {cur_round}".format(cur_round=cur_round))
        accuracy = arena.get_accuracy()
        loss = arena.get_loss()
        age = arena.get_model_ages()
        shielded = arena.get_shields()
        self._log_performance_dict("Accuracy", accuracy, arena, reverse=True)
        self._log_performance_dict("Loss", loss, arena, reverse=False)
        self._log_performance_dict("Ages", age, arena, reverse=True)
        self._log_performance_dict("Shielded", shielded, arena, reverse=True)
        if new_candidates is not None:
            eliminated_model_ids = [
                arena.model_candidates[arena_id].mcm.model_id()
                for arena_id in eliminated]
            new_model_ids = [
                candidate.mcm.model_id()
                for arena_id, candidate in new_candidates.items()]
            self._log("Elimating Models Of Arena ID: {arena_ids}".format(arena_ids=eliminated))
            self._log("Their Model IDs: {model_ids}".format(model_ids=eliminated_model_ids))
            self._log("New Replacing Model IDs: {model_ids}".format(model_ids=new_model_ids))
        else:
            self._log("Eliminated none of the models during this round")

    def log_learning_rate(
        self,
        learning_rate
    ):
        self._log("====Learning Rate Change====")
        self._log("New Learning Rate: {learning_rate}".format(learning_rate=learning_rate))

    def _log_performance_dict(
        self,
        metric_name,
        performance_dict,
        arena,
        reverse=True
    ):
        self._log("====={metric_name}=====".format(metric_name=metric_name))
        arena_ids = list(performance_dict.keys())
        model_ids = {
            arena_id: arena.model_candidates[arena_id].mcm.model_id()
            for arena_id in arena_ids}
        performance_dict = sorted(performance_dict.items(), key=lambda x: x[1], reverse=reverse)
        for arena_id, value in performance_dict:
            self._log("Arena ID: {arena_id} Model ID: ".format(arena_id=arena_id) +
                      "{model_id} {metric_name}: {value}".format(
                          arena_id=arena_id,
                          model_id=model_ids[arena_id],
                          metric_name=metric_name,
                          value=str(value)))
