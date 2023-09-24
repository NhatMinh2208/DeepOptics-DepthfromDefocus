from typing import Tuple
import os
import glob
import torch
import numpy as np
import skimage.io
from torch.utils.data import Dataset
from datasets.augmentation import RandomTransform
from kornia.augmentation import CenterCrop
import cv2
import skimage.transform
from utils.helper import ips_to_metric, metric_to_ips

CROP_WIDTH = 20
TRAIN_BASE_DIR = os.path.join('data', 'training_data', 'dualpixel', 'train')
TEST_BASE_DIR = os.path.join('data', 'training_data', 'dualpixel', 'test')

#enable openEXR
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

def get_captures(base_dir):
    """Gets a list of captures (The IDs)."""
    depth_dir = os.path.join(base_dir, 'inpainted_depth')
    return [
        name for name in os.listdir(depth_dir)
        if os.path.isdir(os.path.join(depth_dir, name))
    ]


class DualPixel(Dataset):
    def __init__(self, dataset: str, image_size: Tuple[int, int], is_training: bool = True, randcrop: bool = False,
                 augment: bool = False, padding: int = 0, upsample_factor: int = 2):
        super().__init__()
        if dataset == 'train':
            base_dir = TRAIN_BASE_DIR
        elif dataset == 'val':
            base_dir = TEST_BASE_DIR
        else:
            raise ValueError(f'dataset ({dataset}) has to be "train," "val," or "example."')
        
        self.transform = RandomTransform(image_size, randcrop, augment) #RandomTransform class instance
        self.centercrop = CenterCrop(image_size)  #Crop a given image tensor at the center.

        captures = get_captures(base_dir)
        self.sample_ids = []
        for id in captures:
            image_path = glob.glob(os.path.join(base_dir, 'scaled_images', id, '*_center.jpg'))[0]
            depth_path = glob.glob(os.path.join(base_dir, 'inpainted_depth', id, '*_center.png'))[0]
            conf_path = glob.glob(os.path.join(base_dir, 'merged_conf', id, '*_center.exr'))[0]
            sample_id = {
                'image_path': image_path,
                'depth_path': depth_path,
                'conf_path': conf_path,
                'id': id,
            }
            self.sample_ids.append(sample_id)
        self.min_depth = 0.2
        self.max_depth = 100.
        self.is_training = is_training
        self.padding = padding
        self.upsample_factor = upsample_factor
        print("===================DualPixel is initialized============================")
	

    def stretch_depth(self, depth, depth_range, min_depth):
        return depth_range * depth + min_depth

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        image_path = sample_id['image_path']
        depth_path = sample_id['depth_path']
        conf_path = sample_id['conf_path']
        id = sample_id['id']

        depth = skimage.io.imread(depth_path).astype(np.float32)[..., None] / 255 #[..., None] used to add a new axis to the NumPy array.
        #Specifically, it adds a new axis of size 1 to the last dimension of the array. 
        #This is often done to convert a 2D image (e.g., grayscale) into a 3D image with a single channel. 
        #It's a common step when working with image data in deep learning frameworks like PyTorch or TensorFlow, 
        #which typically expect images to have a shape of (height, width, channels).
        #it's common when working with grayscale images, where you convert a 2D image with shape (height, width) into a 3D image with shape (height, width, 1)
        img = skimage.io.imread(image_path).astype(np.float32) / 255
        #[...,]: The ellipsis ... represents all the existing dimensions of the array. 
        # In this context, it's used to indicate that we want to keep all the existing dimensions of the array as they are. 
        # It's often used when you don't want to specify the exact dimensions explicitly.
        #[2]: This is an indexing operation. It selects a specific channel from the image.
        #In this case, it's selecting the third channel (assuming the channels are 0-indexed), which could be, 
        #for example, the alpha channel or some other channel of interest.
        #So, in summary, this line of code reads an image using OpenCV and then selects a specific channel from that image. 
        # This is often done when you are working with multi-channel images (e.g., RGBA images) 
        # and want to extract or focus on a specific channel of interest for further processing or analysis.
        conf = cv2.imread(filename=conf_path, flags=-1)[..., [2]]

        depth = depth[CROP_WIDTH:-CROP_WIDTH, CROP_WIDTH:-CROP_WIDTH, :]
        img = img[CROP_WIDTH:-CROP_WIDTH, CROP_WIDTH:-CROP_WIDTH, :]
        conf = conf[CROP_WIDTH:-CROP_WIDTH, CROP_WIDTH:-CROP_WIDTH, :]
        #   this line of code is adding symmetric reflection padding to the image img along its height and width dimensions (top-down, left-right),
        #  while keeping the color channels unchanged (no padding in the color channel dimension).
        #  This can be useful for various image processing tasks,
        #  such as convolutional neural networks (CNNs) where you want to maintain the spatial symmetry of the image during processing.
        #  example:
        #a = [1, 2, 3, 4, 5]
        #np.pad(a, (2, 3), 'reflect')
        #array([3, 2, 1, 2, 3, 4, 5, 4, 3, 2])
        img = np.pad(img,
                     ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='reflect')
        depth = np.pad(depth,
                       ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='reflect')
        conf = np.pad(conf,
                      ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='reflect')

        if self.upsample_factor != 1:
            img = skimage.transform.rescale(img, self.upsample_factor, multichannel=True, order=3)
            depth = skimage.transform.rescale(depth, self.upsample_factor, multichannel=True, order=3)
            conf = skimage.transform.rescale(conf, self.upsample_factor, multichannel=True, order=3)
            #order=3: This parameter specifies the interpolation order to use during rescaling. 
            # An interpolation order of 3 indicates cubic interpolation, which is a commonly used interpolation method for image resizing.
            # Cubic interpolation provides smoother results compared to lower-order interpolations, 
            # but it also requires more computational resources.
        img = torch.from_numpy(img).permute(2, 0, 1)
        depthmap = torch.from_numpy(depth).permute(2, 0, 1)
        conf = torch.from_numpy(conf).permute(2, 0, 1)
        #.permute(2, 0, 1): After converting the NumPy array to a PyTorch tensor, this line of code rearranges the dimensions of the tensor. The permute method is used to reorder the dimensions. In this case:
        #2 represents the current third dimension (e.g., channels in an image). It is moved to the first dimension.
        #0 represents the current first dimension (e.g., height in an image). It is moved to the second dimension.
        #1 represents the current second dimension (e.g., width in an image). It is moved to the third dimension.
        
        
        #? 
        depthmap_metric = ips_to_metric(depthmap, self.min_depth, self.max_depth)
        if depthmap_metric.min() < 1.0:
            depthmap_metric += (1. - depthmap_metric.min())
        depthmap = metric_to_ips(depthmap_metric.clamp(1.0, 5.0), 1.0, 5.0)


        if self.is_training:
            img, depthmap, conf = self.transform(img, depthmap, conf)
        else:
            img = self.centercrop(img)
            depthmap = self.centercrop(depthmap)
            conf = self.centercrop(conf)

        # Remove batch dim (Kornia adds batch dimension automatically.)
        img = img.squeeze(0)
        depthmap = depthmap.squeeze(0)
        depth_conf = torch.where(conf.squeeze(0) > 0.99, 1., 0.)

        sample = {'id': id, 'image': img, 'depthmap': depthmap, 'depth_conf': depth_conf}
        """
        id: string
        image: torch.Size([3, img_sz, img_sz])      [0,1]
        depthmap: torch.Size([1, img_sz, img_sz])      [0,1]
        depth_conf: torch.Size([1, img_sz, img_sz])      0 or 1
        """
        return sample