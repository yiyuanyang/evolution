"""
    Content: Training Driver File for gradient descent
    Author: Yiyuan Yang
    Date: April. 19th 2020
"""

import os
import torch
import copy
import torch.nn as nn
from torch.utils import data
from yyycode.model.models.resnet.resnet import gen_model
from PIL import Image, ImageFilter
import numpy as np


class GradientAscentGenerator(object):
    """
        This is for gradient descent
    """
    def __init__(self, config):
        self.device = torch.device("cuda")
        self.basic_config, self.data_config, self.train_config, self.save_config = \
            config["basic_config"], config["data_config"], \
            config["train_config"], config["save_config"]
        self.encoding = self.data_config["encoding"]
        self.model = gen_model(self.train_config["model_config"]).to(
            self.device)
        cur_model_path = os.path.join(
            self.train_config["model_path"],
            "model_epoch_" + str(self.train_config["epoch"]) + ".pt")
        self.image_save_dir = self.save_config["image_save_dir"]
        self.model.load_state_dict(torch.load(cur_model_path)) 

    def save_image(self, im, cur_class, iteration, prediction, pd_confidence, gt_confidence):
        if not os.path.exists(self.image_save_dir):
            os.mkdir(self.image_save_dir)
        class_save_dir = os.path.join(
            self.image_save_dir,
            str(self.encoding[cur_class]))
        if not os.path.exists(class_save_dir):
            os.mkdir(class_save_dir)
        cur_image_save_dir = os.path.join(
            class_save_dir,
            "iter_{iteration}_pred_{prediction}_conf_{pd_confidence}_{gt_confidence}.jpg".format(
                iteration=iteration,
                prediction=self.encoding[prediction],
                pd_confidence=pd_confidence,
                gt_confidence=gt_confidence))
        if isinstance(im, (np.ndarray, np.generic)):
            im = self.format_np_output(im)
            im = Image.fromarray(im)
        im.save(cur_image_save_dir)

    def train(self):
        # Get variables
        self.learning_config = self.train_config["learning_config"]
        self.learning_rate = self.learning_config["learning_rate"]
        # Start training
        for current_class in range(self.train_config["num_classes"]):
            #Cross Entropy Loss
            #cur_class = torch.from_numpy(
            #    np.array([current_class])).long().to(self.device)
            #MSE Loss
            ground_truth = torch.from_numpy(
                np.array(
                    [[0 if current_class != i else 1 for i in range(self.train_config["num_classes"])]]
                )).float().to(self.device)
            print("Generating for {current_class}".format(current_class=current_class))
            self.generated_image = np.uint8(np.random.uniform(0, 255, (200, 200, 3)))
            self.learning_rate = self.learning_config["learning_rate"]
            for cur_iter in range(0, self.learning_config["num_iterations"]):
                prediction, pd_confidence, gt_confidence = self.iteration(
                    cur_iter, 
                    ground_truth, 
                    current_class)
                if cur_iter % self.save_config["iter_per_save"] == 0:
                    print("Cur Iter: {cur_iter}".format(cur_iter=cur_iter))
                    self.generated_image = self.recreate_image(self.processed_image.cpu())
                    self.save_image(
                        self.generated_image,
                        current_class,
                        cur_iter,
                        prediction,
                        pd_confidence,
                        gt_confidence)

    def recreate_image(self, im_as_var):
        """
            Recreates images from a torch variable, sort of reverse preprocessing
        Args:
            im_as_var (torch variable): Image to recreate
        returns:
            recreated_im (numpy arr): Recreated image in array
        """
        #reverse_mean = [-0.485, -0.456, -0.406]
        #reverse_std = [1/0.229, 1/0.224, 1/0.225]
        recreated_im = copy.copy(im_as_var.data.numpy()[0])
        #for c in range(3):
        #    recreated_im[c] /= reverse_std[c]
        #    recreated_im[c] -= reverse_mean[c]
        #recreated_im[recreated_im > 1] = 1
        #recreated_im[recreated_im < 0] = 0
        recreated_im = (recreated_im - recreated_im.min()) / (recreated_im.max() - recreated_im.min())
        recreated_im = np.round(recreated_im * 255)
        recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
        return recreated_im

    def iteration(
        self,
        cur_iter,
        ground_truth,
        current_class
    ):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        if cur_iter % self.learning_config["blur_freq"] == 0:
            self.processed_image = self.preprocess_and_blur_image(
                self.generated_image,
                self.learning_config["blur_rad"])
        else:
            self.processed_image = self.preprocess_and_blur_image(self.generated_image)

        #optimizer = torch.optim.SGD(
        #    [self.processed_image],
        #    lr=self.learning_rate,
        #    weight_decay=self.learning_config["gamma"])
        #optimizer.zero_grad()
        #prediction = self.model(self.processed_image)
        #loss = loss_func(prediction, ground_truth)
        #loss.backward()

        if self.learning_config["clipping"]:
            torch.nn.utils.clip_grad_norm(
                self.processed_image,
                self.learning_config["clip_value"])
        #loss_func = nn.CrossEntropyLoss()
        loss_func = nn.MSELoss()
        optimizer = torch.optim.LBFGS(
            [self.processed_image],
            lr=self.learning_rate)
        prediction_value = np.array([])
        def closure():
            optimizer.zero_grad()
            prediction = self.model(self.processed_image)
            loss = loss_func(prediction, ground_truth)
            loss.backward()
            return loss

        prediction = self.model(self.processed_image)
        loss = loss_func(prediction, ground_truth)
        optimizer.step(closure)

        # Calculations
        self.learning_rate *= (1 - self.learning_config["gamma"])

        # Format
        prediction_value = prediction.data.cpu().numpy().tolist()
        pred_value_min = np.min(prediction_value[0])
        prediction_value_normalized = [value - pred_value_min for value in prediction_value[0]]
        pred_value_sum = np.sum(prediction_value_normalized)
        pred_confidence = [prob/pred_value_sum for prob in prediction_value_normalized]
        loss = loss.data.cpu().numpy().tolist()
        prediction = [pred.index(max(pred)) for pred in prediction_value][0]
        pd_confidence = int(round(pred_confidence[prediction] * 100))
        gt_confidence = int(round(pred_confidence[current_class] * 100))
        if cur_iter % self.save_config["iter_per_display"] == 0:
            print("pred:{prediction}, gt:{ground_truth}, L:{loss}, lr:{lr}, pd_conf:{pd_confidence}, gt_conf:{gt_confidence}".format(
                prediction=prediction,
                ground_truth=current_class,
                loss=round(loss, 3),
                lr=round(self.learning_rate, 5),
                pd_confidence=pd_confidence,
                gt_confidence=gt_confidence))
        return prediction, pd_confidence, gt_confidence

    def preprocess_and_blur_image(self, pil_im, blur_rad=None):
        """
            Processes image with optional Gaussian blur for CNNs
        Args:
            PIL_img (PIL_img): PIL Image or numpy array to process
            resize_im (bool): Resize to 224 or not
            blur_rad (int): Pixel radius for Gaussian blurring (default = None)
        returns:
            im_as_var (torch variable): Variable that contains processed float tensor
        """
        # mean and std list for channels (Imagenet)
        mean = [0.49719918, 0.49763868, 0.49760074]
        std = [0.229, 0.224, 0.225]

        # ensure or transform incoming image to PIL image
        if type(pil_im) != Image.Image:
            try:
                pil_im = Image.fromarray(pil_im)
            except Exception as _:
                print(
                    "could not transform PIL_img to a PIL Image object. Please check input.")

        # add gaussin blur to image
        if blur_rad:
            pil_im = pil_im.filter(ImageFilter.GaussianBlur(blur_rad))

        im_as_arr = np.float32(pil_im)
        im_as_arr = np.moveaxis(im_as_arr, 2, 0)  # Convert array to D,W,H
        # Normalize the channels
        for channel, _ in enumerate(im_as_arr):
            im_as_arr[channel] /= 255
            im_as_arr[channel] -= mean[channel]
            im_as_arr[channel] /= std[channel]
        # Convert to float tensor
        im_as_ten = torch.from_numpy(im_as_arr).float()
        # Add one more channel to the beginning. Tensor shape = 1,3,224,224
        im_as_ten.unsqueeze_(0)
        # Convert to Pytorch variable
        im_as_var = torch.autograd.Variable(im_as_ten.cuda(), requires_grad=True)
        return im_as_var

    def format_np_output(self, np_arr):
        """
            This is a (kind of) bandaid fix to streamline saving procedure.
            It converts all the outputs to the same format which is 3xWxH
            with using sucecssive if clauses.
        Args:
            im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
        """
        # Phase/Case 1: The np arr only has 2 dimensions
        # Result: Add a dimension at the beginning
        if len(np_arr.shape) == 2:
            np_arr = np.expand_dims(np_arr, axis=0)
        # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
        # Result: Repeat first channel and convert 1xWxH to 3xWxH
        if np_arr.shape[0] == 1:
            np_arr = np.repeat(np_arr, 3, axis=0)
        # Phase/Case 3: Np arr is of shape 3xWxH
        # Result: Convert it to WxHx3 in order to make it saveable by PIL
        if np_arr.shape[0] == 3:
            np_arr = np_arr.transpose(1, 2, 0)
        # Phase/Case 4: NP arr is normalized between 0-1
        # Result: Multiply with 255 and change type to make it saveable by PIL
        if np.max(np_arr) <= 1:
            np_arr = (np_arr*255).astype(np.uint8)
        return np_arr
