"""
    Content: Logger Object that can be passed around to do logging
    Author: Yiyuan Yang
    Date: April. 18th 2020
"""

import os
import sys

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

    def fatal(self, message):
        self._log("FATAL: " + message, force_dump=True)
        sys.exit()
    
    def warning(self, message):
        self._log("WARNING: " + message)

    def log(self, message):
        self._log("LOGGING: " + message)

    def _log(self, log, force_dump = False):
        self.messages.append(log)
        if force_dump or len(self.messages) >= self.dump_frequency:
            self._dump()

    def _dump(self):
        f = open(self.logger_save_dir, "a")
        f.writelines(self.messages)

