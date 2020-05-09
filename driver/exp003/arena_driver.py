"""
    Content: Driver file for initial exp001 arena
    Author: Yiyuan Yang
    Date: April. 30th 2020
"""

import sys
import os
import torch


def set_up_project():
    project_path = \
        "C:\\Users\\yangy\\Documents\\ComputerVision\\Projects\\code\\projects"
    config_path = \
        "Evolution\\config\\experiment_config\\exp003\\arena_config.yml"
    sys.path.append(project_path)
    return os.path.join(project_path, config_path)


def run():
    torch.multiprocessing.freeze_support()


def main():
    run()
    config_path = set_up_project()
    from Evolution.trainer.exp001.arena_trainer import Trainer
    trainer = Trainer(config_path)
    trainer.start_experiment()


if __name__ == "__main__":
    main()
