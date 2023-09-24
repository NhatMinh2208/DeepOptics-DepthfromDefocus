import torch, math
from utils.cubicspline import interp
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
def find_index(a, v):
    a = a.squeeze(1).cpu().numpy()
    v = v.cpu().numpy()
    index = np.stack([np.searchsorted(a[i, :], v[i], side='left') - 1 for i in range(a.shape[0])], axis=0)
    return torch.from_numpy(index)

# c = find_index(a, b)
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

psf = torch.rand((3, 3, 3)).to(device)
#jv1 = scipy.special.jv(1, psf)
jv2 = torch.special.bessel_j1(psf)
#print(jv1)
print(jv2)