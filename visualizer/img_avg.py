import copy
import numpy as np
from PIL import Image

def avg_img(images):
    image_shape = np.asarray(Image.open(images[0])).shape
    img_total = np.zeros(image_shape)
    for image in images:
        cur_image = Image.open(image)
        img_total += np.asarray(cur_image)
    return img_total/len(images)

def visualize_avg(img_avg, save_path=None):
    img_copy = copy.deepcopy(img_avg)
    img_copy = ((img_copy - img_copy.min()) / (img_copy.max()-img_copy.min())) * 255
    img_copy = img_copy.astype(np.uint8)
    pil_img = Image.fromarray(img_copy)
    pil_img.show()
    if save_path:
        pil_img.save(save_path)