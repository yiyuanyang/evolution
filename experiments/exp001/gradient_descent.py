"""
    Content: Driver file for initial exp001
    Author: Yiyuan Yang
    Date: April. 19th 2020
"""

import sys

sys.path.append("C:\\Users\\yangy\\Documents\\ComputerVision\\Projects\\code\\projects")

from Evolution.train.exp001.experiment_preparer import ExperimentPreparer
from Evolution.train.exp001.gradient_descent import Trainer

Exp001Preparer = ExperimentPreparer("config\\experiment_config\\exp001\\gradient_descent.yml")
trainer = Trainer(Exp001Preparer)
trainer.train()