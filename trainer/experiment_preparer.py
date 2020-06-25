"""
    Content: This serves as a general experiment preparator
    It should automatically detect certain parameters of the experiment
    Author: Yiyuan Yang
    Date: April. 18th 2020
"""

import yaml
import os
from Evolution.utils.logger.logger import Logger
from PIL import Image
import pandas as pd
import numpy as np


class ExperimentPreparer(object):
    def __init__(
        self,
        config_path,
    ):
        with open(config_path, "r") as ymlfile:
            config = yaml.load(ymlfile, Loader=yaml.FullLoader)
        self.load_config(config)
        self.process_config()

    def process_config(self):
        self.process_data_config()
        self.process_save_config()
        self.logger = Logger(logger_save_dir=self.save_config["save_dir"],
                             dump_frequency=self.save_config["dump_frequency"])

    def detect_image_params(self):
        train_path_to_labels = pd.read_csv(self.data_config["train_path_to_labels"])
        path = train_path_to_labels.loc[0, "path"]
        im_shape = np.array(Image.open(path)).shape
        if len(im_shape) == 2:
            self.data_config["in_channels"] = 1
        else:
            self.data_config["in_channels"] = im_shape[2]
        assert im_shape[0] == im_shape[1], "Input Images Must Have Equal Width and Height"
        self.data_config["image_size"] = im_shape[0]
        self.train_config["model_config"]["in_channels"] = self.data_config["in_channels"]
        self.train_config["model_config"]["image_size"] = im_shape[0]

    # ** Trivial Helper Functions

    def load_config(self, config):
        self.config = config
        self.basic_config, \
            self.data_config, \
            self.train_config, \
            self.save_config = config

    def create_dir(self, directory):
        if not os.path.exists(directory):
            os.system("mkdir " + directory)

    def print_config(self):
        print(yaml.dump(self.config))

    def get_config(self):
        return self.config

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

    def process_path_to_labels(self):
        self.data_config["train_path_to_labels"] = os.path.join(
            self.data_config["path_to_labels_dir"],
            "train_path_to_labels.csv")
        self.data_config["eval_path_to_labels"] = os.path.join(
            self.data_config["path_to_labels_dir"],
            "eval_path_to_labels.csv")
        self.data_config["test_path_to_labels"] = os.path.join(
            self.data_config["path_to_labels_dir"],
            "test_path_to_labels.csv")
        encodings = pd.read_csv(os.path.join(
            self.data_config["path_to_labels_dir"], 
            "label_to_encoding.csv"))
        encoding_dict = {}
        for _, row in encodings.iterrows():
            encoding_dict[row["class"]] = row["code"]
        self.data_config["encodings"] = encoding_dict

    def get_each_config(self):
        return self.basic_config, self.data_config, self.train_config, self.save_config

    def get_logger(self):
        return self.logger

    def process_data_config(self):
        self.process_path_to_labels()
        self.detect_image_params()
        if "augmentation_config" in self.data_config.keys():
            self.data_config["augmentation_config"]["example_save_dir"] = os.path.join(
                self.save_config["save_dir"],
                "image_processing_examples"
            )

    def process_save_config(self):
        self.save_config["save_dir"] = \
            os.path.join(
                self.save_config["save_dir"],
                self.basic_config["experiment_name"]
            )
        self.save_config["model_save_dir"] = \
            os.path.join(
                self.save_config["save_dir"],
                "models"
            )
        if not os.path.exists(self.save_config["save_dir"]):
            self.create_dir(self.save_config["save_dir"])
        if not os.path.exists(self.save_config["model_save_dir"]):
            self.create_dir(self.save_config["model_save_dir"])
