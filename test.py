import torch, math
from utils.interp import interp
import torch.nn.functional as F
import numpy as np
from utils import complex 
torch.manual_seed(42)
import scipy.special
def matting(depthmap, n_depths, binary, eps=1e-8):
    """
    - depthmap: A PyTorch tensor representing a depth map. The depth values are expected to be in the range [0, 1].
    - n_depths: An integer representing the number of depth levels.
    - binary: A boolean flag indicating whether binary or soft matting should be applied.
    - eps: A small epsilon value to avoid division by zero.
    """
    depthmap = depthmap.clamp(eps, 1.0)
    d = torch.arange(0, n_depths, dtype=depthmap.dtype, device=depthmap.device).reshape(1, 1, -1, 1, 1) + 1
    print("shape of d: ")
    print(d.shape)
    depthmap = depthmap * n_depths
    print("shape of depthmap: ")
    print(depthmap.shape)
    #This computes the absolute difference between the depth values in d and the scaled depthmap
    diff = d - depthmap
    print("shape of diff: ")
    print(diff.shape)
    alpha = torch.zeros_like(diff)
    if binary:
        alpha[torch.logical_and(diff >= 0., diff < 1.)] = 1.
    else:
        mask = torch.logical_and(diff > -1., diff <= 0.)
        alpha[mask] = diff[mask] + 1.
        alpha[torch.logical_and(diff > 0., diff <= 1.)] = 1.
    return alpha
# Assuming img is a depthmap of size 20x20
# img = torch.rand(5, 1,360, 360)  # Example random depthmap
# img = img[:, None, ...]
# # Calculate matting with 7 depth levels and binary matting
# result = matting(img, 7, True)

# img2 = torch.rand(5, 3, 360, 360)
# result = result * img2[:, :, None, ...]

# # Print the shape of the result
# print(result)
# print(result.shape)


# A = torch.rand(5, 6, 1, 7) 
# B = torch.rand(5, 1, 7, 8)  
# C = torch.matmul(A, B)
# print(C.shape) #5, 6, 1, 8 -> (W, D, 1, P)
# print(C.squeeze(-2).shape) # 5, 6, 8 -> (W, D, P)


#A = torch.rand(5, 6, 1, 7)

# image_size = [60, 60]
# print(image_size[1])
# A = torch.arange(1, image_size[1] // 2 + 1)
# B = torch.arange(-1, max(image_size) // 2 + 1, dtype=torch.double) + 0.5
# print(60 // 2  + 1)
# print(torch.arange(1, 30))
# print(A.shape)
# print(B.shape)


# a = torch.rand(3, 8)
# b = torch.rand(3, 6, 6)
# def find_index(a, v):
#     a = a.squeeze(1).cpu().numpy()
#     v = v.cpu().numpy()
#     index = np.stack([np.searchsorted(a[i, :], v[i], side='left') - 1 for i in range(a.shape[0])], axis=0)
#     return torch.from_numpy(index)
# a[0][0] = 0.1052
# c = find_index(a, b)

# print(a)
# print( b)
# print(c)
# print(c.shape)

# shape = 12
# a = torch.rand(3, 5, 8)
# wl = torch.rand(3)
# b = math.sqrt(2) * (torch.arange(-1, 12 // 2 + 1, dtype=torch.double) + 0.5)
# c1 = torch.arange(1, 12 // 2 + 1).reshape(-1, 1)
# c2 = torch.arange(1, 12 // 2 + 1).reshape(1, -1)
# print(torch.sqrt(c1 ** 2 + c2 ** 2))
# c = torch.sqrt(c1 ** 2 + c2 ** 2).unsqueeze(0) / wl.reshape(-1, 1, 1)
# print(b)
# b = b.reshape(1, 1, -1) / wl.reshape(-1, 1, 1)
# b = b.squeeze(1)
# d = find_index(b,c)
# print(d)
# print(b.shape)
# print(c.shape)
# # print(d.shape)
# # n, h, w = d.shape
# # print(torch.arange(n))
# # #e = a[0, :, d[0] + 1]
# # e = d[0] + 1
# # print(e.shape)
# #e = torch.stack([a[i, :, d[i] + 1] for i in torch.arange(n)], dim=0)
# e = F.relu(interp(b, a, c, d).float())
# e = e.reshape(3, 5, 12 // 2, 12 // 2)

# print(e)
# print(e.shape)
# n_wl X D X img // 2 X img // 2



# b = torch.rand(3, 8)
# c = torch.rand(3, 6, 6)
# c = c.numpy()
# print(c[0].shape)
# print(b[0, :].shape)

# d = np.searchsorted(b[0, :], c[0], side='left')
# print(d.shape)

# Create a 3D tensor (a 2x3x4 tensor)
# tensor = torch.tensor([[[1, 2, 3, 4],
#                         [5, 6, 7, 8],
#                         [9, 10, 11, 12]],
#                        [[13, 14, 15, 16],
#                         [17, 18, 19, 20],
#                         [21, 22, 23, 24]]])

# Calculate the sum along the last two dimensions with keepdims=True
# sum_tensor = tensor.sum(dim=(-2, -1), keepdims=True)

# print("Original Tensor:")
# print(tensor)
# print(tensor.shape)
# print("\nSum along last two dimensions with keepdims=True:")
# print(sum_tensor)
# print(sum_tensor.shape)

# div = tensor / sum_tensor
# print("\nDiv:")
# print(div)
# print(div.shape)

# Set a random seed for reproducibility
# torch.manual_seed(42)

# # Create a tensor of shape (3, 5, 8, 8) with random values
# # https://stackoverflow.com/questions/71515439/equivalent-to-torch-rfft-in-newest-pytorch-version
# tensor = torch.rand((3, 5, 8, 8))
# #fft = torch.view_as_real(torch.fft.rfft2(tensor))
# fft = torch.fft.rfft2(tensor)

# print(fft)
# print(fft.shape)

def autocorrelation1d_symmetric(h):
    """Compute autocorrelation of a symmetric signal along the last dimension"""
    #Fhsq = complex.abs2(torch.rfft(h, 1))
    Fhsq = complex.abs2(torch.view_as_real(torch.fft.rfft(h)))
    #a = torch.irfft(torch.stack([Fhsq, torch.zeros_like(Fhsq)], dim=-1), 1, signal_sizes=(h.shape[-1],))
    a = torch.fft.irfft(torch.view_as_complex(torch.stack([Fhsq, torch.zeros_like(Fhsq)], dim=-1)), h.size(-1))
    return a / a.max()

def compute_weighting_for_tapering(h):
    # h:  B, C, H, W
    """Compute autocorrelation of a symmetric signal along the last two dimension"""
    h_proj0 = h.sum(dim=-2, keepdims=False) #B, C, W
    autocorr_h_proj0 = autocorrelation1d_symmetric(h_proj0).unsqueeze(-2) #B, C, 1,  W
    h_proj1 = h.sum(dim=-1, keepdims=False) #B, C, 1
    autocorr_h_proj1 = autocorrelation1d_symmetric(h_proj1).unsqueeze(-1) #B, C, 1, 1
    return (1 - autocorr_h_proj0) * (1 - autocorr_h_proj1)
# psf = torch.rand((4, 3, 5, 8, 8)) # B, C, D, H, W
# psf = psf.mean(dim=-3) #[4, 3, 8, 8]
# # #print(psf.shape)
# #psf = psf.sum(dim=-2, keepdims=False) #[4, 3, 8])
# # #psf = torch.view_as_real(torch.fft.rfft(psf))
# # wtf = (psf.shape[-1],)

# psf = psf.sum(dim=-2, keepdims=False) #B, C, W
# psf = autocorrelation1d_symmetric(psf).unsqueeze(-2)
# psf = psf.sum(dim=-1, keepdims=False) #B, C, 1
# #psf = autocorrelation1d_symmetric(psf).unsqueeze(-1)
# h = psf
# psf = complex.abs2(torch.view_as_real(torch.fft.rfft(psf))) #B, C, 1
# psf = torch.stack([psf, torch.zeros_like(psf)], dim=-1) #B, C, 1, 2
# psf =  torch.view_as_complex(psf) #B, C, 1
# psf = torch.fft.irfft(psf, h.size(-1))
# print(psf.shape)

# psf = torch.rand((4, 3, 5, 8, 8)) # B, C, D, H, W
# psf = psf.mean(dim=-3) #[4, 3, 8, 8]
# psf = compute_weighting_for_tapering(psf)
# psf1 = psf
# #print(psf1.shape) # 4, 3, 8, 8
# psf2 = torch.view_as_real(torch.fft.rfft2(psf))
# #psf=  complex.multiply(psf2, psf2) #([4, 3, 8, 5, 2])
# #psf2 = torch.view_as_complex(complex.multiply(psf2, psf2)) # 4, 3, 8, 5
# #psf = torch.fft.irfft2(psf2, psf1.size())
# psf = torch.fft.irfft2(torch.view_as_complex(complex.multiply(psf2, psf2)), psf1.shape[-2:])
# print(psf)
# print(psf.shape)

# t = torch.rand(4, 3, 8, 8)
# T = torch.fft.rfft2(t)
# print(T.shape)
# roundtrip = torch.fft.irfft2(T, t.shape[-2:])
# print(roundtrip.shape)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# psf = torch.rand((3, 3, 3)).to(device)
# #jv1 = scipy.special.jv(1, psf)
# jv2 = torch.special.bessel_j1(psf)
# #print(jv1)
# print(jv2)


# tensor2x2 = torch.rand((4, 4))
# tensor2x2 = (tensor2x2 > 0.5).float()
# tensor4x4 = F.interpolate(tensor2x2.unsqueeze(0).unsqueeze(0), size=(8, 8), mode='nearest').squeeze()
# print(tensor2x2)
# print(tensor2x2.shape)
# print(tensor4x4)
# print(tensor4x4.shape)

# tensor = torch.ones((3, 4, 4))
# tensor2 = torch.rand((4,4)) 
# tensor3 = tensor * tensor2
# print(tensor3)
# print(tensor3.shape)

# def to_sensor_phase(size, size1, wavelengths):
#         x_prime = size * torch.arange(1, size).reshape(-1, 1, 1, 1)
#         y_prime = size * torch.arange(1, size).reshape(1, -1, 1, 1)
#         x = size * torch.arange(1, size1).reshape(1, 1, -1, 1)
#         y = size * torch.arange(1, size1).reshape(1, 1, 1, -1)
#         distance_diff = (x - x_prime) ** 2 + (y - y_prime) **2
#         distance_diff = distance_diff.unsqueeze(0)
#         z = 5
#         k = (2 * math.pi) / wavelengths.reshape(-1, 1, 1, 1, 1)
#         return k * distance_diff / (2 * z)

# size = 5
# size1 = 7
# wavelengths = torch.rand([3])
# sth = to_sensor_phase(size, size1, wavelengths)
# print(sth)
# print(sth.shape)

# def find_index(a, v):
#     a = a.squeeze(1).cpu().numpy()
#     v = v.cpu().numpy()
#     index = np.stack([np.searchsorted(a[i, :], v[i], side='left') - 1 for i in range(a.shape[0])], axis=0)
#     return torch.from_numpy(index)

# shape = 12
# a = torch.rand(3, 5, 8)
# wl = torch.rand(3)
# b = math.sqrt(2) * (torch.arange(-1, 12 // 2 + 1, dtype=torch.double) + 0.5)
# c1 = torch.arange(1, 12 // 2 + 1).reshape(-1, 1)
# c2 = torch.arange(1, 12 // 2 + 1).reshape(1, -1)
# print(torch.sqrt(c1 ** 2 + c2 ** 2))
# # c = torch.sqrt(c1 ** 2 + c2 ** 2).unsqueeze(0) / wl.reshape(-1, 1, 1)
# print(b)
# # b = b.reshape(1, 1, -1) / wl.reshape(-1, 1, 1)
# # b = b.squeeze(1)
# b = b.unsqueeze(0)
# c = torch.sqrt(c1 ** 2 + c2 ** 2).unsqueeze(0)
# d = find_index(b,c)
# print(d)
# print(b.shape)
# print(c.shape)
# e = F.relu(interp(b, a, c, d).float())
# print(e.shape)
# e = e.reshape(3, 5, 12 // 2, 12 // 2)
# print(e.shape)

# Define the dimensions
# A = 2
# B = 3
# C = 4
# D = 5
# E = 6
# #torch.Size([3, 16, 1, 16])
# #torch.Size([3, 1, 16, 82])
# # Create tensors with shapes (A, B, C) and (B, C, D)
# tensor1 = torch.randn(A, 1, C, D, E)
# tensor2 = torch.randn(A, B, 1, C)

# # Perform matrix multiplication
# result = torch.matmul(tensor2, tensor1)

# # Print the shape of the result tensor
# print(result.shape)

# torch.Size([32, 1, 1, 1])
# torch.Size([1, 1, 31, 1])
# torch.Size([1, 32, 32, 31, 31])
# torch.Size([3, 1, 16])
# torch.Size([3, 16, 16])
# phase:
# torch.Size([1, 16, 1, 32, 32])
# in_camera_phase:
# torch.Size([3, 1, 32, 32, 31, 31])
# tensor = torch.randn(3, 3, 5, 5, 6, 6)
# x = tensor.size(2) * tensor.size(3)
# tensor = tensor.reshape(tensor.size(0), tensor.size(1), -1, tensor.size(4), tensor.size(5))
# tensor = tensor.reshape(tensor.size(0), tensor.size(1), x, -1)
# print(tensor.shape)

# mask_size = 100
# tensor = torch.arange(1, mask_size)
# print(tensor.shape)


f_number = 6.3
focal_length = 50e-3
focal_depth = 1.7
s = 1. / (1. / focal_length - 1. / focal_depth)
condition = 0.01
pixel_pitch = 6.45e-6

sth = math.sqrt(condition * 4 / f_number) * s / pixel_pitch

print(sth)

img_size = 256 + 32 * 4
#pixel_pitch = 6.45e-6
image_length = img_size * pixel_pitch
mask_size = 256
mask_diameter = focal_length / f_number
mask_length = mask_size * mask_diameter
print("image_length: ")
print(image_length)   #0.0024768
print("mask_length: ")
print(mask_diameter)  #0.007936507936507938
print("mask_pitch: ") #0.000031001984126984
print(mask_diameter / mask_size)
print("sensor2camera") #0.05151515151515152
print("wave_length")  #[632e-9, 550e-9, 450e-9]
print("mask_init_size") #3.87e-05
print(image_length / 64)
print(3.87e-05 * 64 / 384)


# sensor 
# diameter: 0.0024768
# resolution : 384
# pitch : 6.45e-6

# mask
# diameter: 0.0024768
# resolution : 256
# pitch: 



