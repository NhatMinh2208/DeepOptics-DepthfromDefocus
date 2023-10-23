import torch
import torch.nn.functional as F
mask_size = 12
sensor_size = 100
mask_pitch = 1
pixel_pitch = 1

r_coord = torch.arange(1, mask_size + 1) - 0.5
rho_coord = torch.arange(1, sensor_size + 1)
h = 23 * r_coord + 3
depth = torch.arange(1, 17)
wl = torch.arange(1, 4)
h = h.unsqueeze(0) / depth.reshape(-1, 1)
h = h.unsqueeze(0) / wl.reshape(-1, 1 ,1)
print(r_coord.shape)
print("shape: ")
print(h.shape[-1])

# def linterp(r_coord, h, rho_coord):
a = rho_coord * pixel_pitch / mask_pitch
diff = a - torch.floor(a)
alpha = torch.where(diff <= 0.5, diff + 0.5, diff - 0.5)
index = torch.floor(a - 0.5).to(torch.int)
mask1 = index < 0 
mask2 = index == (h.shape[-1] - 1)
index[mask1] = 0
# b = torch.floor(a).to(torch.int) - 1
# print(len(h))
# print(index)
# print(mask1)
desired_size = torch.max(index)
# Calculate the padding size for the 1D tensor
padding = max(0, desired_size - h.shape[-1] + 2)
alb = F.pad(h, (0, padding))


# mask = b < len(h)
# alb = torch.where(mask , h[b], 0)

# print(b)
# print(alb.shape)

result = (1 - alpha) * alb[:, :, index] + alpha * alb[:, :, index + 1]
result[:, :, mask1] = h[:, :, 0].unsqueeze(-1)
result[:, :, mask2] = h[:, :, -1].unsqueeze(-1)
print(result.shape)




