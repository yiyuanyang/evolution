"""
    Content: Logger Object that can be passed around to do logging
    Author: Yiyuan Yang
    Date: April. 18th 2020
"""

import os
import sys
import yaml
from sklearn import metrics

class Logger(object):
    def __init__(
        self, 
        logger_save_dir, 
        dump_frequency = 10
    ):
        self.messages = []
        self.dump_frequency = 10
        self.phase = 3

        if not os.path.exists(logger_save_dir):
            print("FATAL: Save Directory Does not Exist")
            sys.exit()
        
        self.logger_save_dir = [
            os.path.join(logger_save_dir, "train.log",
            os.path.join(logger_save_dir, "eval.log",
            os.path.join(logger_save_dir, "test.log",
            os.path.join(logger_save_dir, "initial.log")
        ]


    # Basic helper functions
    def _log(self, log, force_dump = False, print_to_console=True):
        self.messages.append(log)
        if force_dump or len(self.messages) >= self.dump_frequency:
            self._dump()
        if print_to_console:
            print(log)

    def set_phase(
        self,
        epoch,
        phase,
        print_to_console
    ):
        """
            Set current phase of logger
        """
        if len(self.messages) != 0:
            self._dump()
        self.phase = phase

        log = "LOGGING: Starting phase " + str(phase) + " for epoch " + str(epoch)
        self._log(
            log=log,
            print_to_console=print_to_console
        )


    def _dump(self):
        """
            This saves logs to a file
        """
        file = open(self.logger_save_dir[self.phase], "a")
        file.writelines(self.messages)

    # Here are all the random logging functions

    def fatal(
        self, 
        message, 
        print_to_console=True
    ):
        self._log(
            log="FATAL: " + message, 
            force_dump=True, 
            print_to_console=print_to_console
        )
        sys.exit()
    
    def warning(
        self, 
        message, 
        print_to_console=True
    ):
        self._log(
            log="WARNING: " + message, 
            print_to_console=print_to_console
        )

    def log(
        self, 
        message,
        print_to_console=True
    ):
        self._log(
            log="LOGGING: " + message,
            print_to_console=print_to_console
        )

    def log_learning_rate_change(
        self, 
        epoch, 
        cur, 
        new, 
        print_to_console=True
    ):
        """
            Logs changes in learning rate
        """
        log = ("LOGGING: Epoch " + str(epoch) + " , adjusted learning rate from " + 
            str(cur) + " to " + str(new))
    
        self._log(
            log=log,
            print_to_console=print_to_console
        )

    def log_config(
        self, 
        config_name, 
        config, 
        print_to_console=True
    ):
        """
            This logs a config
        """
        log = "LOGGING: " + config_name + ": " + yaml.dump(config)

        self._log(
            log=log,
            print_to_console=print_to_console
        )

    def log_batch_result(
        self, 
        batch_index,
        prediction,
        ground_truth,
        loss,
        print_to_console=True
    ):
        """
            This logs a batch run's result
        """
        log = ("LOGGING: Batch " + str(batch_index) + "\n Prediction:" +
            str(prediction) + "\n Ground Truth: " + str(ground_truth) + "\n Loss: " + str(loss))

        self._log(
            log=log, 
            print_to_console=print_to_console
        )

    def log_data(
        self,
        batch_index,
        data,
        print_to_console=True
    ):
        """
            This logs the shape and value of data
        """
        log = ("LOGGING: Batch" + str(batch_index) +
                " Data Shape: " + str(list(data.shape))+ 
                " Data Value: " + str(data.cpu().numpy.tolist()))
        
        self._log(
            log=log,
            print_to_console=print_to_console
        )
    
    def _log_accuracy(
        self,
        epoch,
        ground_truth,
        prediction,
        print_to_console=True
    ):  
        """
            Logs the confusion matrix
        """
        accuracy_score = metrics.accuracy_score(ground_truth, prediction)
        log = ("LOGGING: Epoch " + str(epoch) + 
                " Accuracy Score: " + str(accuracy_score))
        
        self._log(
            log=log,
            print_to_console=print_to_console
        )

    def _log_f1(
        self,
        epoch,
        ground_truth,
        prediction,
        print_to_console=True
    ):  
        """
            Logs the confusion matrix
        """
        accuracy_score = metrics.f1_score(ground_truth, prediction)
        log = ("LOGGING: Epoch " + str(epoch) + 
                " f1 Score: " + str(accuracy_score))
        
        self._log(
            log=log,
            print_to_console=print_to_console
        )

    def _log_recall(
        self,
        epoch,
        ground_truth,
        prediction,
        print_to_console=True
    ):  
        """
            Logs the confusion matrix
        """
        accuracy_score = metrics.recall_score(ground_truth, prediction)
        log = ("LOGGING: Epoch " + str(epoch) + 
                " recall Score: " + str(accuracy_score))
        
        self._log(
            log=log,
            print_to_console=print_to_console
        )

    def _log_precision(
        self,
        epoch,
        ground_truth,
        prediction,
        print_to_console=True
    ):  
        """
            Logs the confusion matrix
        """
        accuracy_score = metrics.precision_score(ground_truth, prediction)
        log = ("LOGGING: Epoch " + str(epoch) + 
                " precision Score: " + str(accuracy_score))
        
        self._log(
            log=log,
            print_to_console=print_to_console
        )

    def _log_confusion_matrix(
        self,
        epoch,
        ground_truth,
        prediction,
        print_to_console=True
    ):  
        """
            Logs the confusion matrix
        """
        confusion_matrix = metrics.confusion_matrix(
            ground_truth,
            prediction
        )
        log = ("LOGGING: Epoch " + str(epoch) + 
                "\n Confusion Matrix: \n" + str(confusion_matrix))
        
        self._log(
            log=log,
            print_to_console=print_to_console
        )

    def log_epoch_metrics(
        self,
        epoch,
        ground_truth,
        prediction,
        print_to_console=True
    ):
        """
            Perform all metrics
        """
        ground_truth = [int(item) for item in ground_truth]
        prediction = [int(item) for item in ground_truth]
        self.log("===================EPOCH SUMMARY=====================",print_to_console)
        self._log_accuracy(epoch,ground_truth,predictionprint_to_console)
        self._log_recall(epoch,ground_truth,prediction,print_to_console)
        self._log_precision(epoch,ground_truth,prediction,print_to_console)
        self._log_f1(epoch,ground_truth,predction,print_to_console)
        self._log_confusion_matrix(epoch,ground_truth,prediction,print_to_console)

