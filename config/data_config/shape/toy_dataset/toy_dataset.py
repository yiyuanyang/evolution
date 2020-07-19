from torch.utils.data import Dataset
from torchvision import transforms as tf
import pandas as pd
from PIL import Image
import numpy as np


class ToyShapeDataset(Dataset):
    """
        Dataset for toy set
    """

    def __init__(
        self,
        file_dir,
        augmentation=True,
        im_size=128
    ):
        self.file_dir = file_dir
        self.augmentation = augmentation
        self.im_size = im_size
        path_to_label = pd.read_csv(file_dir)
        self.data = []
        for _, row in path_to_label.iterrows():
            path = row['path']
            file_name = path.split("\\")[-1].split(".")[0]
            pil_im = Image.open(path)
            pil_im = pil_im.resize((self.im_size, self.im_size))
            label = row['label']
            self.data.append([pil_im, label, file_name])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pil_im, label, file_name = self.data[idx]
        if self.augmentation:
            pil_im = self.simple_augment(pil_im)
        im = np.array(pil_im).astype(np.float32).reshape(
            (1, self.im_size, self.im_size))
        norm_im = self.normalize_image(im)
        inv_norm_im = self.invert_image(norm_im)
        return [inv_norm_im, label, file_name]

    def normalize_image(self, im):
        """
            Normalize The Image Between -1 and 1
        """
        return ((im - np.min(im)) / (np.max(im) - np.min(im))) - 1

    def invert_image(self, im):
        """
            Make black as background and white as brush stroke
        """
        return -im

    def simple_augment(self, pil_im):
        degrees = 45
        translate = (0.1, 0.1)
        scale = None
        shear = None
        augment = tf.RandomAffine(
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear
        )
        return augment(pil_im)
