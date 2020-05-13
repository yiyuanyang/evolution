"""
    Content: Driver file for initial exp001 gradient descent
    Author: Yiyuan Yang
    Date: April. 19th 2020
"""

import sys
import os
import torch


def set_up_project():
    project_path = "C:\\Users\\yangy\\Documents\\ComputerVision\\Projects\\code\\projects"
    config_path = "Evolution\\config\\experiment_config\\exp003\\gradient_descent_config.yml"
    sys.path.append(project_path)
    return os.path.join(project_path, config_path)


def run():
    torch.multiprocessing.freeze_support()


def main():
    run()
    config_path = set_up_project()
    from Evolution.trainer.exp001.experiment_preparer import ExperimentPreparer
    from Evolution.trainer.exp001.gradient_descent import Trainer

    Exp001Preparer = ExperimentPreparer(config_path)
    trainer = Trainer(Exp001Preparer)
    trainer.train()


if __name__ == "__main__":
    main()
