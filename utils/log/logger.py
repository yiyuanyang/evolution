"""
    Content: Logger Object that can be passed around to do logging
    Author: Yiyuan Yang
    Date: April. 18th 2020
"""

import os
import sys
import yaml

class Logger(object):
    def __init__(
        self, 
        logger_save_dir, 
        dump_frequency = 10
    ):
        self.messages = []
        self.dump_frequency = 10

        if not os.path.exists(logger_save_dir):
            print("FATAL: Save Directory Does not Exist")
            sys.exit()
        self.logger_save_dir = logger_save_dir


    # Basic helper functions
    def _log(self, log, force_dump = False, print_to_console=True):
        self.messages.append(log)
        if force_dump or len(self.messages) >= self.dump_frequency:
            self._dump()
        if print_to_console:
            print(log)

    def _dump(self):
        """
            This saves logs to a file
        """
        f = open(self.logger_save_dir, "a")
        f.writelines(self.messages)

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

