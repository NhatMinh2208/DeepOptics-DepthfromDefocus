import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
im = cv2.imread('./psf_mag.tif', cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
tensor = torch.from_numpy(im)
print(tensor.shape)
plt.imshow(im)
plt.show()