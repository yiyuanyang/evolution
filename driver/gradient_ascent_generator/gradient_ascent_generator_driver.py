"""
    Content: Driver file for initial exp001 gradient descent
    Author: Yiyuan Yang
    Date: April. 19th 2020
"""

import sys
import os
import torch
import yaml

PROJECT_PATH = "C:\\Users\\yangy\\Documents\\ComputerVision\\Projects\\"
CONFIG_PATH = "yyycode\\config\\exp_config"

def set_up_project():
    exp_name = "basic_shapes"
    config_path = os.path.join(CONFIG_PATH, exp_name, "gradient_ascent_config.yml")
    sys.path.append(PROJECT_PATH)
    return os.path.join(PROJECT_PATH, config_path)


def run():
    torch.multiprocessing.freeze_support()


def main():
    run()
    config_path = set_up_project()
    from yyycode.trainer.basic_shapes.gradient_ascent_generator import GradientAscentGenerator
    with open(config_path, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    trainer = GradientAscentGenerator(config)
    trainer.train()


if __name__ == "__main__":
    main()
