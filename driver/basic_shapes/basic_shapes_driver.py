"""
    Content: Driver file for initial exp001 gradient descent
    Author: Yiyuan Yang
    Date: April. 19th 2020
"""

import sys
import os
import torch

PROJECT_PATH = "C:\\Users\\yangy\\Documents\\ComputerVision\\Projects\\"
CONFIG_PATH = "yyycode\\config\\exp_config"

def set_up_project():
    exp_name = "basic_shapes"
    config_path = os.path.join(CONFIG_PATH, exp_name, exp_name + "_config.yml")
    sys.path.append(PROJECT_PATH)
    return os.path.join(PROJECT_PATH, config_path)


def run():
    torch.multiprocessing.freeze_support()


def main():
    run()
    config_path = set_up_project()
    from yyycode.trainer.experiment_preparer import ExperimentPreparer
    from yyycode.trainer.basic_shapes.basic_shapes_trainer import Trainer

    Exp001Preparer = ExperimentPreparer(config_path)
    trainer = Trainer(Exp001Preparer)
    trainer.train()


if __name__ == "__main__":
    main()
