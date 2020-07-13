from torch.utils.data import Dataset
from torchvision import transforms as tf
import torch
from yyycode.utils.data.augmentation.image_augmentation import ImageAugmentor
import pandas as pd
from PIL import Image
import numpy as np


class BasicShapesDataset(Dataset):
    """
        Dataset for toy set
    """

    def __init__(
        self,
        file_dir,
        augmentation_config=None,
        grayscale=False,
        im_size=200
    ):
        self.file_dir = file_dir
        self.grayscale = grayscale
        if augmentation_config is not None:
            self.augment = True
            self.image_augmentor = ImageAugmentor(augmentation_config, grayscale=grayscale)
        else:
            self.augment = False
        self.im_size = im_size
        path_to_label = pd.read_csv(file_dir)
        self.data = []
        for index, row in path_to_label.iterrows():
            path = row['path']
            pil_im = Image.open(path)
            label = row['label']
            pil_im = pil_im.resize((self.im_size, self.im_size))
            self.data.append([np.array(pil_im), label])
            if index % 1000 == 0:
                print("Read Image {cur_id}/{total}".format(cur_id=index, total=path_to_label.shape[0]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.augment:
            image = self.image_augmentor.augment_image(image)
        np_image = np.asarray(image)
        if not self.grayscale:
            return np.moveaxis(np_image, 2, 0)
        return (image, label)

    def normalize_image(self, im):
        """
            Normalize The Image Between -1 and 1
        """
        return ((im - np.min(im)) / (np.max(im) - np.min(im))) - 1
