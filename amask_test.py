import torch
# https://stackoverflow.com/questions/72325827/how-to-require-gradient-only-for-some-tensor-elements-in-a-pytorch-tensor

def circular_mask(shape):
    radius = shape // 2
    Y, X = torch.meshgrid(torch.arange(radius), torch.arange(radius))
    mask = (X)**2 + (Y)**2 <= (radius - 1)**2
    return mask.float()

def copy_quadruple(x_rd):
    x_ld = torch.flip(x_rd, dims=(-2,))
    x_d = torch.cat([x_ld, x_rd], dim=-2)
    x_u = torch.flip(x_d, dims=(-1,))
    x = torch.cat([x_u, x_d], dim=-1)
    return x

tensor = torch.randn(8, 8)
print(copy_quadruple(circular_mask(8)))
