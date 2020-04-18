"""
  Content: Dataset for CIFAR-10
  Author: Yiyuan Yang
  Date: April 7th 2020
"""

import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import os

class CIFAR10Dataset(Dataset):
    """
        Datset for CIFAR 10
    """

    def __init__(self, data_dir, batches, image_size):
        self.data_dir = data_dir
        self.image_size = image_size
        self.file_names = []
        self.data_dict = {}
        for batch in batches:
            cur_file_names, cur_data_dict = self._process_batch(os.path.join(data_dir, batch))
            self.file_names += cur_file_names
            self.data_dict.update(cur_data_dict)
        
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file_name = self.file_names[idx]
        return self.data_dict[file_name]

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
        return np.stack([red, green, blue])
        