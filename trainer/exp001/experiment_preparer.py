"""
    Content: Reading Configs for exp001
    Author: Yiyuan Yang
    Date: April. 18th 2020
"""

import yaml
import os
from Evolution.utils.logger.logger import Logger


class ExperimentPreparer(object):
    def __init__(
        self,
        config_path,
    ):
        with open(config_path, "r") as ymlfile:
            self.config = yaml.load(ymlfile, Loader=yaml.FullLoader)
        self.process_config()

    def process_config(self):
        basic_config, data_config, _, save_config = self.get_each_config()
        data_config = self.process_data_path(0, data_config)
        data_config = self.process_data_path(1, data_config)
        data_config = self.process_data_path(2, data_config)
        save_config["save_dir"] = \
            os.path.join(
                save_config["save_dir"],
                basic_config["experiment_name"]
            )
        if "augmentation_config" in data_config.keys():
            data_config["augmentation_config"]["example_save_dir"] = os.path.join(
                save_config["save_dir"],
                "image_processing_examples"
            )
        save_config["model_save_dir"] = \
            os.path.join(
                save_config["save_dir"],
                "models"
            )
        self.create_dir(save_config["save_dir"])
        self.create_dir(save_config["model_save_dir"])
        self.set_each_config(data_config=data_config, save_config=save_config)
        self.logger = Logger(logger_save_dir=save_config["save_dir"],
                             dump_frequency=save_config["dump_frequency"])

    def create_dir(self, directory):
        if not os.path.exists(directory):
            os.system("mkdir " + directory)

    def print_config(self):
        print(yaml.dump(self.config))

    def get_config(self):
        return self.config

    def get_each_config(self):
        basic_config = self.config["basic_config"]
        data_config = self.config["data_config"]
        train_config = self.config["train_config"]
        save_config = self.config["save_config"]
        return basic_config, data_config, train_config, save_config

    def set_each_config(self,
                        basic_config=None,
                        data_config=None,
                        train_config=None,
                        save_config=None):
        if basic_config:
            self.config["basic_config"] = basic_config
        if data_config:
            self.config["data_config"] = data_config
        if train_config:
            self.config["train_config"] = train_config
        if save_config:
            self.config["save_config"] = save_config

    def process_data_path(self, phase, data_config):
        all_phases = ["train_data", "eval_data", "test_data"]
        cur_phase = all_phases[phase]
        for i in range(len(data_config[cur_phase])):
            data_config[cur_phase][i] = os.path.join(data_config["data_dir"],
                                                     data_config[cur_phase][i])
        return data_config

    def get_logger(self):
        return self.logger
