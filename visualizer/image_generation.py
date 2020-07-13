"""
    Content:In this file, we use gradient descent on images
    and observe what the netowrk generates
    Author: Yiyuan Yang
    Date: May 13th 2020
"""

import torch
from torch import nn
from yyycode.model.models.resnet.resnet import gen_model
from torchvision import models
import numpy as np
import os
from PIL import Image


class ImageGenerator(object):
    def __init__(self, config):
        self.config = config
        self.model_dir = config["model_dir"]
        self.model_config = config["model_config"]
        self.image_shape = [10] + config["image_shape"]
        self.save_dir = config["save_dir"]
        self.initial_learning_rate = config["learning_rate"]
        self.regularizer = config["regularizer"]
        self.iterations_per_save = config["iterations_per_save"]
        if not os.path.exists(self.save_dir):
            os.system("mkdir " + self.save_dir)

    def init_model_optimizer(self, image):
        #self.model = gen_model(self.model_config).to("cuda")
        #self.model.load_state_dict(torch.load(self.model_dir))
        self.model = models.resnet34(pretrained=True).to("cuda")
        self.optim = torch.optim.LBFGS([image.requires_grad_()])
        for param in self.model.parameters():
            param.requires_grad = False

    def init_noise_tensor(self):
        gray_image = torch.ones(self.image_shape).to("cuda") * 128
        noise = torch.randn(self.image_shape).to("cuda")
        gray_image = (gray_image + noise)
        #gray_image = (gray_image - gray_image.min()) / (gray_image.max() - gray_image.min())
        return gray_image

    def init_trained_image(self, class_type, iteration):
        self.image_save_dir = os.path.join(
            self.save_dir,
            str(class_type),
            "iter_{iteration}.jpg".format(iteration=iteration))
        image = np.asarray(Image.open(self.image_save_dir)).astype(np.float32)
        image = np.moveaxis(self.normalize_image(image), 2, 0)
        return image

    def init_training_tensor(self, iteration):
        arrays = []
        for i in range(10):
            arrays.append(self.init_trained_image(i, iteration))
        numpy_batch = np.stack(arrays)
        tensor_image = torch.tensor(numpy_batch).to("cuda")
        return tensor_image

    def regularize_image(self, image):
        return image - self.regularizer * np.absolute(
            np.multiply(image, image)
        )

    def normalize_image(self, image):
        for index, channel in enumerate(image):
            channel_max = np.amax(channel)
            channel_min = np.amin(channel)
            channel = ((channel - channel_min) / (channel_max - channel_min)) * 255
            channel = channel.round().astype(np.uint8)
            image[index] = channel
        return image.astype(np.uint8)

    def save_image(self, pil_image, class_type, iteration):
        image_save_dir = os.path.join(
            self.save_dir,
            str(class_type))
        if not os.path.exists(image_save_dir):
            os.system("mkdir " + image_save_dir)
        image_save_dir = os.path.join(
            image_save_dir,
            "iter_{iteration}.jpg".format(iteration=iteration))
        pil_image.save(image_save_dir)

    def save_tensor(self, batch_tensor, iteration):
        batch_numpy = batch_tensor.data.cpu().numpy()
        for i in range(10):
            numpy_image = batch_numpy[i]
            numpy_image = self.regularize_image(numpy_image)
            numpy_image = self.normalize_image(numpy_image)
            numpy_image = np.moveaxis(numpy_image, 0, 2)
            pil_image = Image.fromarray(numpy_image)
            self.save_image(pil_image, i, iteration)

    def iteration(self, image, label, iteration):
        if iteration % self.iterations_per_save == 0:
            self.save_tensor(image, iteration)
            print("Regularizer Term {term}".format(
                term=self.regularizer * (image ** 2).mean())
            )
        def closure():
            self.optim.zero_grad()
            prediction = self.model(image)
            losses = []
            for i in range(10):
                losses.append(
                    -prediction[i][i] + self.regularizer * (image ** 2).mean())
            loss = torch.mean(torch.stack(losses))
            if loss.requires_grad:
                loss.backward()
            print(loss)
            return loss
        self.optim.step(closure)

    def reverse_training(self):
        starting_iteration = self.config["starting_iteration"]
        max_iterations = self.config["max_iterations"]
        label = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).to("cuda")
        if self.config["use_existing_picture"]:
            image = self.init_training_tensor(starting_iteration)
        else:
            image = self.init_noise_tensor()
        self.init_model_optimizer(image)
        self.model.eval()
        for iteration in range(starting_iteration, max_iterations):
            print("Iteration {cur_iter}/{total_iter}".format(
                cur_iter=iteration,
                total_iter=max_iterations
            ))
            self.iteration(image, label, iteration)
