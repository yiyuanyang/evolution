"""
    Content: This file drives the visualization of trained models
    generating images
    Author: Yiyuan Yang
    Date: May 11th 2020
"""


import sys
import torch


config = {
    "model_dir": "E:\\saved_experiments\\exp003\\" +
                 "exp003_arena_resnet_18_image_generation\\model_140.pt",
    "model_config": {
        "model_type": "resnet18",
        "in_channels": 3,
        "image_size": 32,
        "num_classes": 10,
        "kernel_sizes": [7, 3, 3, 3, 3],
    },
    "starting_iteration": 0,
    "use_existing_picture": False,
    "max_iterations": 1000,
    "iterations_per_save": 10,
    "learning_rate": 0.1,
    "regularizer": 0.25,
    "image_shape": [3, 256, 256],
    "image_class": 1,
    "save_dir": "E:\\saved_experiments\\exp003\\" +
                "exp003_arena_resnet_18_image_generation\\generated_images"
}


def set_up_project():
    project_path = \
        "C:\\Users\\yangy\\Documents\\ComputerVision\\Projects\\code\\projects"
    sys.path.append(project_path)


def run():
    torch.multiprocessing.freeze_support()


def main():
    run()
    set_up_project()
    from Evolution.visualizer.image_generation import ImageGenerator
    image_generator = ImageGenerator(config)
    image_generator.reverse_training()


if __name__ == "__main__":
    main()
