# import torch

# def matting(depthmap, n_depths, binary, eps=1e-8):
#     """
#     - depthmap: A PyTorch tensor representing a depth map. The depth values are expected to be in the range [0, 1].
#     - n_depths: An integer representing the number of depth levels.
#     - binary: A boolean flag indicating whether binary or soft matting should be applied.
#     - eps: A small epsilon value to avoid division by zero.
#     """
#     depthmap = depthmap.clamp(eps, 1.0)
#     d = torch.arange(0, n_depths, dtype=depthmap.dtype, device=depthmap.device).reshape(1, 1, -1, 1, 1) + 1
#     print("shape of d: ")
#     print(d.shape)
#     depthmap = depthmap * n_depths
#     print("shape of depthmap: ")
#     print(depthmap.shape)
#     #This computes the absolute difference between the depth values in d and the scaled depthmap
#     diff = d - depthmap
#     print("shape of diff: ")
#     print(diff.shape)
#     alpha = torch.zeros_like(diff)
#     if binary:
#         alpha[torch.logical_and(diff >= 0., diff < 1.)] = 1.
#     else:
#         mask = torch.logical_and(diff > -1., diff <= 0.)
#         alpha[mask] = diff[mask] + 1.
#         alpha[torch.logical_and(diff > 0., diff <= 1.)] = 1.
#     return alpha
# # Assuming img is a depthmap of size 20x20
# img = torch.rand(20, 20)  # Example random depthmap
# img = img[:, None, ...]
# # Calculate matting with 7 depth levels and binary matting
# result = matting(img, 7, True)

# # Print the shape of the result
# print(result.shape)
from datasets.dualpixel import DualPixel
from datasets.sceneflow import SceneFlow
import torch.utils.data

image_sz = 200
crop_width = 40
augment = True
randcrop = True
padding = 0
sf_train_dataset = SceneFlow('train',
                                (image_sz + 4 * crop_width,
                                image_sz + 4 * crop_width),
                                is_training=True,
                                randcrop=randcrop, augment=augment, padding=padding,
                                singleplane=False)

dp_train_dataset = DualPixel('train',
                                (image_sz + 4 * crop_width,
                                image_sz + 4 * crop_width),
                                is_training=True,
                                randcrop=randcrop, augment=augment, padding=padding)

train_dataset = torch.utils.data.ConcatDataset([dp_train_dataset, sf_train_dataset])

a = dp_train_dataset[0]
print(a['image'].shape)
print(a['depthmap'].shape)
print(a['depth_conf'].shape)



# n_sf = len(sf_train_dataset)
# n_dp = len(dp_train_dataset)
# print("n_sf=",n_sf)
# print("n_dp=", n_dp)
# sample_weights = torch.cat([1. / n_dp * torch.ones(n_dp, dtype=torch.double),
#                             1. / n_sf * torch.ones(n_sf, dtype=torch.double)], dim=0)
# sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
# train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=sampler,
#                                 num_workers=1, shuffle=False, pin_memory=True)



