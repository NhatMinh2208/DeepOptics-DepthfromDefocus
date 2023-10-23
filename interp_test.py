import torch
import numpy as np
import math 
# from utils.interp import linterp
x = torch.rand((3, 8))
y = torch.rand((3, 5, 8))
xs = torch.rand((3, 6, 6))

def find_index1(a, v):
    #The find_index function you've provided takes two tensors, a and v, and returns a tensor of indices where 
    #each index corresponds to the leftmost position in a where the corresponding element in v can be inserted
    #while maintaining the sorted order.
    a = a.cpu().numpy()
    v = v.cpu().numpy()
    index = np.searchsorted(a[:], v, side='left') - 1
    return torch.from_numpy(index)

mask_pitch = 1
mask_size = 20
wl = [632e-9, 550e-9, 450e-9]
wl = torch.Tensor(wl)
depth = torch.arange(1, 16)

coord_y = mask_pitch * (torch.arange(1, mask_size // 2 + 1) - 0.5).reshape(-1, 1)
coord_x = mask_pitch * (torch.arange(1, mask_size // 2 + 1) - 0.5).reshape(1, -1)
coord_y = coord_y.double()
coord_x = coord_x.double()
# (mask_size / 2) x (mask_size / 2)
r_sampling = torch.sqrt(coord_y ** 2 + coord_x ** 2)
#  print(r_sampling)

r_grid = math.sqrt(2) * mask_pitch * (
        torch.arange(1, mask_size // 2 + 1, dtype=torch.double) - 0.5) # 10 

# ind = find_index(r_grid, r_sampling) # (10, 10)
# ind[0,0] = 0
# print(ind)
# print(ind.shape)
# diff = (r_sampling - r_grid[ind]) / (r_grid[ind + 1] - r_grid[ind])

y = torch.rand_like(r_grid)
y = y.unsqueeze(0) / wl.reshape(-1, 1)
y = y.unsqueeze(1) / depth.reshape(1, -1, 1)
# temp = y[0]
# result = y[ind] * (1 - diff) + y[ind + 1] * diff
# result[0,0] = temp
# print(result)
# print(result.shape)
# result = linterp(r_grid, y, r_sampling)
# print(result.shape)
# result = torch.rand(3,16, 256)
# result = torch.nn.functional.interpolate(result, 8, mode='linear')
# print(result.shape)
def find_index( a, v):
        #The find_index function you've provided takes two tensors, a and v, and returns a tensor of indices where 
        #each index corresponds to the leftmost position in a where the corresponding element in v can be inserted
        #while maintaining the sorted order.
        a = a.squeeze(1).cpu().numpy()
        v = v.cpu().numpy()
        index = np.stack([np.searchsorted(a[i, :], v[i], side='left') - 1 for i in range(a.shape[0])], axis=0)
        return torch.from_numpy(index)
ind = find_index1(r_grid, r_sampling)

def linterp(r_grid, y, r_sampling, index):
    '''
    x: inputs\n
    y: outputs\n
    sample: samples \n
    x must have size larger than 2 to run successfully\n
    -->return: output of samples\n
    '''
    n, _, _ = y.shape # usually n_wl
    index[0,0] = 0
    diff = (r_sampling - r_grid[index]) / (r_grid[index + 1] - r_grid[index]) # (10, 10)
#   diff = torch.stack([(sample[i] - x[i, index[i]]) / (x[i, index[i] + 1] - x[i, index[i]]) for i in torch.arange(n)], dim=0) 
    temp = y[:,:,0]
    sth = y[1, :, index[1]]
    print(sth.shape)
    print(diff.shape)
    result = torch.stack([ y[i, :, index] * (1 - diff) + y[i, :, index + 1] * diff  for i in torch.arange(n)], dim=0) 
    result[:,:,0,0] = temp
    return result

# print(r_grid.shape)  # 10
# print(r_sampling.shape) # (10, 10)
# print(ind.shape) # (10, 10)
# result = linterp(r_grid, y, r_sampling, ind)
# print(result.shape)

# print(wl.shape[0])
# print(depth.shape[0])
# a = torch.zeros((3, 16))
# print(a.shape)

# dx = x[:, [1]] - x[:, [0]] # (3, 1)
# # different betwwen sampling position and sampling index
# # xs[0]: (6, 6) ||  ind[0] : (6, 6)  || x[i, ind[i]] : (6, 6)
# diff = torch.stack([xs[i] - x[i, ind[i]] for i in torch.arange(n)], dim=0) # (3, 6, 6)

# x = x.unsqueeze(1)
# m = (y[..., 1:] - y[..., :-1]) / (x[..., 1:] - x[..., :-1]) / dx.reshape(-1, 1, 1) # (3, 5, 7)
# m = torch.cat([m[..., [0]], (m[..., 1:] + m[..., :-1]) / 2, m[..., [-1]]], dim=-1) # (3, 5, 8)

# y0 = torch.stack([y[i, :, ind[i]] for i in torch.arange(n)], dim=0) # (3, 5, 6, 6)
# print(y0.shape)
# y1 = torch.stack([y[i, :, ind[i] + 1] for i in torch.arange(n)], dim=0)
# print(y1.shape)
# m0 = torch.stack([m[i, :, ind[i]] for i in torch.arange(n)], dim=0)
# print(m0.shape)
# m1 = torch.stack([m[i, :, ind[i] + 1] for i in torch.arange(n)], dim=0)
# print(m1.shape)


# print("xs[0]: ")
# print(xs[0].shape)
# print("ind[0]: ")
# print(ind[0].shape)
# print("x[0, ind[0]]: ")
# print(x[0, ind[0]].shape)

# print("dx: ")
# print(dx.shape) # (3, 1)

# print("diff: ")
# print(diff.shape) # (3, 6, 6)


# new sampling technique:
def mask2sensor_scale(self, value):
    sensor_coord_ = torch.arange(1, self.image_size[0] // 2 + 1).to(self.device)
    r_coord = sensor_coord_ * self.camera_pixel_pitch / self.mask_pitch
    diff = r_coord - torch.floor(r_coord)
    alpha = torch.where(diff <= 0.5, diff + 0.5, diff - 0.5)
    index = torch.floor(r_coord - 0.5).to(torch.int)
    mask1 = index < 0 
    mask2 = index == (value.shape[-1] - 1)
    index[mask1] = 0

    desired_size = torch.max(index)
    # Calculate the padding size for the 1D tensor
    padding = max(0, desired_size - value.shape[-1] + 2)
    pad_value = F.pad(value, (0, padding))

    result = (1 - alpha) * pad_value[:, :, index] + alpha * pad_value[:, :, index + 1]
    result[:, :, mask1] = value[:, :, 0].unsqueeze(-1)
    result[:, :, mask2] = value[:, :, -1].unsqueeze(-1)
    return result



def linterp2(r_grid, y, r_sampling, index):
    '''
    x: inputs\n
    y: outputs\n
    sample: samples \n
    x must have size larger than 2 to run successfully\n
    -->return: output of samples\n
    '''
    mask2 = index >= (r_grid.shape[-1] - 1)
    index[mask2] = 0
    n, _, _ = y.shape # usually n_wl
    index[0,0] = 0
    diff = (r_sampling - r_grid[index]) / (r_grid[index + 1] - r_grid[index]) # (10, 10)
#   diff = torch.stack([(sample[i] - x[i, index[i]]) / (x[i, index[i] + 1] - x[i, index[i]]) for i in torch.arange(n)], dim=0) 
    temp = y[:,:,0]
    sth = y[1, :, index[1]]
    print(sth.shape)
    print(diff.shape)
    result = torch.stack([ y[i, :, index] * (1 - diff) + y[i, :, index + 1] * diff  for i in torch.arange(n)], dim=0) 
    result[:,:,0,0] = temp
    result[:,:,mask2] = 0
    return result

r_grid2 =  mask_pitch * (torch.arange(1, mask_size // 2 + 1, dtype=torch.double) - 0.5) # 10 
ind2 = find_index1(r_grid2, r_sampling)
print(ind2)
print(r_grid2.shape)
y2 = torch.rand_like(r_grid2)
y2 = y2.unsqueeze(0) / wl.reshape(-1, 1)
y2 = y2.unsqueeze(1) / depth.reshape(1, -1, 1)
result = linterp2(r_grid2, y2, r_sampling, ind2)
print(result.shape)
