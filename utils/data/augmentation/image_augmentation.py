"""
    Content: ImageAugmentation class handles all image augmentation processes
    Author: Yiyuan Yang
    Date: April. 26th 2020
"""

from torchvision import transforms as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os


class ImageAugmentor(object):

    def __init__(self, augmentation_config, grayscale=False):
        self.grayscale = grayscale
        self.all_pil_processes = []
        self.all_array_processes = []
        self.example_save_dir = augmentation_config["example_save_dir"]
        if not os.path.exists(self.example_save_dir):
            os.system("mkdir " + self.example_save_dir)
        self.num_examples = augmentation_config["num_examples"]
        self.count = 0

        if augmentation_config is not None:
            self.config = augmentation_config
        else:
            self.config = {}

        # Modify Brightness and/or Contrast and/or Saturation and/or
        if "ColorJitter" in self.config.keys():
            color_jitter_config = self.config["ColorJitter"]
            brightness = color_jitter_config["brightness"]
            contrast = color_jitter_config["contrast"]
            saturation = color_jitter_config["saturation"]
            hue = color_jitter_config["hue"]
            self.all_pil_processes.append(
                tf.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue
                )
            )

        # Add grayscale versions
        if "RandomGrayscale" in self.config.keys():
            self.all_pil_processes.append(
                tf.RandomGrayscale(
                    p=self.config["RandomGrayscale"]["probability"]
                )
            )

        if "RandomHorizontalFlip" in self.config.keys():
            self.all_pil_processes.append(
                tf.RandomHorizontalFlip(
                    p=self.config["RandomHorizontalFlip"]["probability"]
                )
            )

        if "RandomVerticalFlip" in self.config.keys():
            self.all_pil_processes.append(
                tf.RandomVerticalFlip(
                    p=self.config["RandomVerticalFlip"]["probability"]
                )
            )

        # Rotate, Translate, Scale, Shear
        if "RandomAffine" in self.config.keys():
            random_affine_config = self.config["RandomAffine"]
            degrees = random_affine_config["degrees"]
            translate = random_affine_config["translate"]
            scale = random_affine_config["scale"]
            shear = random_affine_config["shear"]
            self.all_pil_processes.append(
                tf.RandomAffine(
                    degrees=degrees,
                    translate=translate,
                    scale=scale,
                    shear=shear
                )
            )

    def save_int_array_CHW(self, image, name):
        self.save_int_array_HWC(np.moveaxis(image, 0, 2), name)

    def save_int_array_HWC(self, image, name):
        if self.grayscale:
            image = Image.fromarray(image, 'L')
        else:
            image = Image.fromarray(image)
        image.save(os.path.join(self.example_save_dir, name))

    def save_pil_image(self, image, name):
        image.save(os.path.join(self.example_save_dir, name))

    def save_float_array_CHW(self, image, name):
        plt.imsave(os.path.join(self.example_save_dir, name), image)

    def save_examples(self, original, augmented):
        assert self.example_save_dir != "", "Example Saving Directory Should Be Set"
        if self.count < self.num_examples:
            self.save_int_array_HWC(original, str(self.count) + "_original.png")
            self.save_pil_image(augmented, str(self.count) + "_augmented.png")
            self.count += 1

    def augment_image(self, image):
        if self.grayscale:
            augmented_image = Image.fromarray(image, 'L')
        else:
            augmented_image = Image.fromarray(image)
        for process in self.all_pil_processes:
            augmented_image = process(augmented_image)
        self.save_examples(image, augmented_image)
        return augmented_image
