"""
  Content: Dataset for CIFAR-10
  Author: Yiyuan Yang
  Date: April 7th 2020
"""

import torch
import torchvision
from torch.utils.data import Dataset
from Evolution.data.data_augmentation.image_augmentation import ImageAugmentor
from PIL import Image
import pickle
import numpy as np
import os

class CIFAR10Dataset(Dataset):
    """
        Datset for CIFAR 10
    """

    def __init__(self, data_dir_list, augmentation_config = None, image_size = 32):
        self.data_dir_list = data_dir_list
        self.image_augmentation = ImageAugmentor(augmentation_config)
        self.image_size = image_size
        self.file_names = []
        self.data_dict = {}
        for data_dir in data_dir_list:
            cur_file_names, cur_data_dict = self._process_batch(data_dir)
            self.file_names += cur_file_names
            self.data_dict.update(cur_data_dict)
        

    def __len__(self):
        return len(self.file_names)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file_name = self.file_names[idx]
        data_tuple = self.data_dict[file_name]
        return (np.moveaxis(np.asarray(self.image_augmentation.augment_image(data_tuple[0])), 2, 0), data_tuple[1]) 


    def _process_batch(self, data_dir):
        """
        Process a batch of data for CIFAR 10
        Args:
            data_dir(string): directory of the file 
        Ret:
            ret_dict(dict(string,tuple(string, np.ndarray))):
                A map between filename and its image label pair
        """
        with open(data_dir, 'rb') as fo:
            data_bundle = pickle.load(fo, encoding='bytes')

        # Obtain the fields
        labels = data_bundle[b'labels']
        data = data_bundle[b'data']
        file_names = data_bundle[b'filenames']
        file_names = [file_name.decode('ascii') for file_name in file_names]

        # Process the data
        data_dict = {}
        for index, name in enumerate(file_names):
            cur_data = self._decode_image(data[index])
            cur_label = labels[index]
            data_dict[name] = (cur_data, cur_label)
        return file_names, data_dict


    def _decode_image(self, row_image):
        """
        Given a row of numbers, convert it back to a 32 * 32, 3 channel image
        Args:
            row_image(nparray): A list of integers of length 3072
        """
        length = int(self.image_size ** 2)
        red = row_image[:length].reshape(32,32)
        green = row_image[length:length * 2].reshape(32,32)
        blue = row_image[length * 2:].reshape(32,32)
        return Image.fromarray(np.stack([red, green, blue], axis = 2))