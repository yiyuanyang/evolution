"""
    Content: Logger object that can be passed around to do logging
    Author: Yiyuan Yang
    Date: April. 18th 2020
"""

import os
import sys
import yaml
from sklearn import metrics
import pandas as pd
import numpy as np

class Logger(object):
    def __init__(
        self, 
        logger_save_dir, 
        dump_frequency = 10,
        print_to_console = True
    ):
        self.messages = []
        self.dump_frequency = 10
        self.phase = 3
        self.print_to_console = print_to_console

        if not os.path.exists(logger_save_dir):
            print("FATAL: Save Directory Does not Exist")
            sys.exit()
        
        self.logger_save_dir = [
            os.path.join(logger_save_dir, "train.log"),
            os.path.join(logger_save_dir, "eval.log"),
            os.path.join(logger_save_dir, "test.log"),
            os.path.join(logger_save_dir, "initial.log")
        ]
        self.stat_save_dir = [
            os.path.join(logger_save_dir, "train.csv"),
            os.path.join(logger_save_dir, "eval.csv"),
            os.path.join(logger_save_dir, "test.csv")
        ]
        self.stats = [
            {
                "epoch":[],
                "global_accuracy": [],
                "accuracy":[],
                "global_recall":[],
                "recall":[],
                "global_precision":[],
                "precision":[],
                "global_f1":[],
                "f1":[]
            },
            {
                "epoch":[],
                "global_accuracy": [],
                "accuracy":[],
                "global_recall":[],
                "recall":[],
                "global_precision":[],
                "precision":[],
                "global_f1":[],
                "f1":[]
            },
            {
                "epoch":[],
                "global_accuracy": [],
                "accuracy":[],
                "global_recall":[],
                "recall":[],
                "global_precision":[],
                "precision":[],
                "global_f1":[],
                "f1":[]
            },
        ]


    # Basic helper functions
    def _log(self, log, force_dump = False):
        self.messages.append(log)
        if force_dump or len(self.messages) >= self.dump_frequency:
            self._dump()
        if self.print_to_console:
            print(log)


    def set_phase(
        self,
        epoch,
        phase
    ):
        """
            Set current phase of logger
        """
        if len(self.messages) != 0:
            self._dump()
        
        if phase != 2:
            self.print_to_console=True

        phases = ["Train", "Eval", "Test"]
        self.phase = phase

        self.log(
            "======================SWITCHING PHASE=========================")
        log = "LOGGING: Starting " + phases[phase] + " phase for epoch " + str(epoch)

        self._log(
            log=log
        )

        if phase == 2:
            self.print_to_console=False


    def _dump(self):
        """
            This saves logs to a file
        """
        file = open(self.logger_save_dir[self.phase], "a")
        file.writelines(self.messages)


    # Here are all the random logging functions
    def fatal(
        self, 
        message
    ):
        self._log(log="FATAL: " + message, force_dump=True)
        sys.exit()
    

    def warning(
        self, 
        message
    ):
        self._log(log="WARNING: " + message)


    def log(self, message):
        self._log(
            log="LOGGING: " + message
        )


    def log_learning_rate_change(self, epoch, cur, new):
        """
            Logs changes in learning rate
        """
        log = ("LOGGING: Epoch " + str(epoch) + " , adjusted learning rate from " + 
            str(cur) + " to " + str(new))
    
        self._log(
            log=log
        )

    def log_config(self, config_name, config):
        """
            This logs a config
        """
        log = "LOGGING: " + config_name + ": " + yaml.dump(config)

        self._log(
            log=log
        )


    #=====================================================
    # Data related logging
    def log_batch_result(
        self, 
        batch_index,
        total_batches,
        prediction_prob,
        prediction,
        ground_truth,
        loss
    ):
        """
            This logs a batch run's result
        """
        prediction_prob = [self.round_to_4_decimal(item) for item in prediction_prob]
        prediction_string = [str(item) for item in prediction_prob]
        prediction_string = "\n".join(prediction_string)
        log = ("LOGGING: Batch " + str(batch_index) + "/" + str(total_batches) +
            "\n Prediction Probability: \n" + prediction_string +
            "\n Prediction:" + str(prediction) + "\n Ground Truth: " + str(ground_truth) + 
            "\n Loss: " + str(loss))

        self._log(
            log=log
        )


    def log_data(self, batch_index, data, label):
        """
            This logs the shape and value of data
        """
        log = ("LOGGING: Batch " + str(batch_index) +
                " Data Shape: " + str(list(data.shape)) + 
                " Label Shape: " + str(list(label.shape)))
        
        self._log(
            log=log
        )


    #======================================================================
    # Performance related logging 
    def _log_accuracy(
        self,
        epoch,
        ground_truth,
        prediction
    ):  
        """
            Logs the confusion matrix
        """
        global_accuracy = metrics.accuracy_score(ground_truth, prediction)

        accuracy_scores = []
        for i in range(0, max(ground_truth)+1):
            label = [element for element in ground_truth if element == i]
            pred = [prediction[j] for j in range(len(ground_truth)) if ground_truth[j] == i]
            accuracy_scores.append(metrics.accuracy_score(label, pred))
        accuracy_scores = self.round_to_2_decimal(accuracy_scores)

        log = ("LOGGING: Epoch " + str(epoch) + 
        " Global Accuracy : " + str(global_accuracy) + 
        "\nAccuracy Scores : " + str(accuracy_scores))
        
        self._log(
            log=log
        )

        return round(global_accuracy,2), accuracy_scores


    def _log_f1(self, epoch, ground_truth, prediction):  
        """
            Logs the confusion matrix
        """

        f1_scores = metrics.f1_score(ground_truth, prediction, average=None)
        global_f1 = round(np.mean(f1_scores),2)
        f1_scores = self.round_to_2_decimal(f1_scores)

        log = ("LOGGING: Epoch " + str(epoch) + 
                " Global F1: " + str(global_f1) +
                "\nF1 Scores: " + str(f1_scores))
        
        self._log(log=log)

        return global_f1, f1_scores


    def _log_recall(self, epoch, ground_truth, prediction):  
        """
            Logs the confusion matrix
        """
        recall_scores = metrics.recall_score(ground_truth, prediction, average=None)
        global_recall = round(np.mean(recall_scores),2)
        recall_scores = self.round_to_2_decimal(recall_scores)
        log = ("LOGGING: Epoch " + str(epoch) +
                " Globall Recall: " + str(global_recall) + 
                "\nRecall Scores: " + str(recall_scores))
        
        self._log(log=log)

        return global_recall, recall_scores


    def _log_precision(self, epoch, ground_truth, prediction):  
        """
            Logs the confusion matrix
        """
        precision_scores = metrics.precision_score(ground_truth, prediction, average=None)
        global_precision = np.mean(precision_scores)
        precision_scores = self.round_to_2_decimal(precision_scores)
        log = ("LOGGING: Epoch " + str(epoch) + 
                " Global Precision: " + str(global_precision) +
                "\nPrecision Scores: " + str(precision_scores))
        
        self._log(log=log)

        return global_precision, precision_scores


    def _log_confusion_matrix(self,epoch,ground_truth,prediction):  
        """
            Logs the confusion matrix
        """
        confusion_matrix = metrics.confusion_matrix(
            ground_truth,
            prediction
        )
        log = ("LOGGING: Epoch " + str(epoch) + 
                "\n Confusion Matrix: \n" + str(confusion_matrix))
        
        self._log(log=log)


    def log_epoch_metrics(self, epoch, ground_truth, prediction, loss):
        """
            Perform all metrics
        """
        ground_truth = [int(item) for item in ground_truth]
        prediction = [int(item) for item in prediction]

        self.log("===================EPOCH SUMMARY=====================")

        global_accuracy, accuracy = self._log_accuracy(
            epoch,
            ground_truth,
            prediction
        )
        global_recall, recall = self._log_recall(
            epoch,
            ground_truth,
            prediction
        )
        global_precision, precision = self._log_precision(
            epoch,
            ground_truth,
            prediction
        )
        global_f1, f1 = self._log_f1(
            epoch,
            ground_truth,
            prediction
        )
        self._log_confusion_matrix(
            epoch,
            ground_truth,
            prediction
        )

        self.stats[self.phase]["epoch"].append(epoch)
        self.stats[self.phase]["global_accuracy"].append(global_accuracy)
        self.stats[self.phase]["accuracy"].append(accuracy)
        self.stats[self.phase]["global_recall"].append(global_recall)
        self.stats[self.phase]["recall"].append(recall)
        self.stats[self.phase]["global_precision"].append(global_precision)
        self.stats[self.phase]["precision"].append(precision)
        self.stats[self.phase]["global_f1"].append(global_f1)
        self.stats[self.phase]["f1"].append(f1)

        stats_dataframe = pd.DataFrame.from_dict(self.stats[self.phase])
        stats_dataframe.to_csv(self.stat_save_dir[self.phase])

    # Model Related Logging
    def log_model(
        self,
        model
    ):
        model_log = str(model)
        self.log("===================")
        self.log("Current Model Used:")
        self.log(model_log)

    def log_model_statistics(
        self,
        model,
        model_name,
        calculate_statistics
    ):
        self._log("\n=====\nWeight Statistics for Model")
        calculate_statistics(model, model_name, self)


    def log_conv_statistics(
        self,
        statistics,
        name
    ):
        weight_statistics, bias_statistics, grad_statistics = statistics
        self._log("Stats for {name}".format(name=name))
        self.log_tensor_statistics(
            weight_statistics, 
            "weight"
        )
        if bias_statistics is not None:
            self.log_tensor_statistics(
                bias_statistics, 
                "bias  "
            )
        if grad_statistics is not None:
            self.log_tensor_statistics(
                grad_statistics, 
                "grad  "
            )


    def log_tensor_statistics(
        self,
        statistics,
        name
    ):
        max_val, min_val, range_val, mean_val, stdev = statistics
        msg = "{name}".format(name=name)
        msg += " max {max_val} min {min_val} range {range_val} mean {mean_val} stdev {stdev}".format(
                max_val=np.format_float_scientific(max_val.data.cpu().numpy(), precision=2),
                min_val=np.format_float_scientific(min_val.data.cpu().numpy(), precision=2),
                range_val=np.format_float_scientific(range_val.data.cpu().numpy(), precision=2),
                mean_val=np.format_float_scientific(mean_val.data.cpu().numpy(), precision=2),
                stdev=np.format_float_scientific(stdev.data.cpu().numpy(), precision=2)
        )
        self._log(msg)


    def round_to_4_decimal(
        self, 
        stats
    ):
        return [round(item, 4) for item in stats]

    def round_to_2_decimal(
        self, 
        stats
    ):
        return [round(item, 2) for item in stats]





