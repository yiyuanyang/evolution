"""
    Content: ImageAugmentation class handles all image augmentation processes
    Author: Yiyuan Yang
    Date: April. 26th 2020
"""

from torchvision import transforms as tf

class ImageAugmentor(object):

    def __init__(self, augmentation_config):
        self.all_processes = []
        
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
            self.all_processes.append(
                tf.ColorJitter(
                    brightness=brightness, 
                    contrast=contrast, 
                    saturation=saturation, 
                    hue=hue
                )
            )

        # Add grayscale versions
        if "RandomGrayscale" in self.config.keys():
            self.all_processes.append(
                tf.RandomGrayscale(
                    p=self.config["RandomGrayscale"]["probability"]
                )
            )

        if "RandomHorizontalFlip" in self.config.keys():
            self.all_processes.append(
                tf.RandomHorizontalFlip(
                    p=self.config["RandomHorizontalFlip"]["probability"]
                )
            )

        if "RandomVerticalFlip" in self.config.keys():
            self.all_processes.append(
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
            self.all_processes.append(
                tf.RandomAffine(
                    degrees=degrees,
                    translate=translate,
                    scale=scale,
                    shear=shear
                )
            )

    def augment_image(self, image):
        for process in self.all_processes:
            image = process(image)
        return image

