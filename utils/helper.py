import torch
import math
import torch.nn.functional as F

def ips_to_metric(d, min_depth, max_depth):
    """
    https://github.com/fyu/tiny/blob/4572a056fd92696a3a970c2cffd3ba1dae0b8ea0/src/sweep_planes.cc#L204

    Args:
        d: inverse perspective sampling [0, 1]
        min_depth: in meter
        max_depth: in meter

    Returns:

    """
    return (max_depth * min_depth) / (max_depth - (max_depth - min_depth) * d)


def metric_to_ips(d, min_depth, max_depth):
    """

    Args:
        d: metric depth [min_depth, max_depth]
        min_dpeth: in meter
        max_depth: in meter

    Returns:
    """
    # d = d.clamp(min_depth, max_depth)
    return (max_depth * d - max_depth * min_depth) / ((max_depth - min_depth) * d)

def matting(depthmap, n_depths, binary, eps=1e-8):
    """
    - depthmap: A PyTorch tensor representing a depth map. The depth values are expected to be in the range [0, 1].
    - n_depths: An integer representing the number of depth levels.
    - binary: A boolean flag indicating whether binary or soft matting should be applied.
    - eps: A small epsilon value to avoid division by zero.
    """
    depthmap = depthmap.clamp(eps, 1.0)
    d = torch.arange(0, n_depths, dtype=depthmap.dtype, device=depthmap.device).reshape(1, 1, -1, 1, 1) + 1
    depthmap = depthmap * n_depths
    #This computes the absolute difference between the depth values in d and the scaled depthmap
    diff = d - depthmap
    alpha = torch.zeros_like(diff)
    if binary:
        alpha[torch.logical_and(diff >= 0., diff < 1.)] = 1.
    else:
        mask = torch.logical_and(diff > -1., diff <= 0.)
        alpha[mask] = diff[mask] + 1.
        alpha[torch.logical_and(diff > 0., diff <= 1.)] = 1.
    return alpha

def depthmap_to_layereddepth(depthmap, n_depths, binary=False):
    """
    output: (B, C, D, H, W)
    - B: Batch size
    - C: Color channel (=1)
    - D: Depth level
    - H, W: Height and Width
    \nis a (normalized || binary) mask at each depth level

    """
    depthmap = depthmap[:, None, ...]  # add color dim in the index number 1 (2nd)
    layered_depth = matting(depthmap, n_depths, binary=binary)
    return layered_depth

def heightmap_to_phase(height, wavelength, refractive_index):
    return height * (2 * math.pi / wavelength) * (refractive_index - 1)


def phase_to_heightmap(phase, wavelength, refractive_index):
    return phase / (2 * math.pi / wavelength) / (refractive_index - 1)

def refractive_index(wavelength, a=1.5375, b=0.00829045, c=-0.000211046):
    """Cauchy's equation - dispersion formula
    Default coefficients are for NOA61.
    https://refractiveindex.info/?shelf=other&book=Optical_adhesives&page=Norland_NOA61
    """
    return a + b / (wavelength * 1e6) ** 2 + c / (wavelength * 1e6) ** 4

def copy_quadruple(x_rd):
    """
    Original Tensor (x_rd):\n
tensor([[[[1., 2.],\n
          [3., 4.]]],\n
        [[[5., 6.],\n
          [7., 8.]]]])\n

Copied Tensor (x):\n
tensor([[[[1., 2., 1., 2.],\n
          [3., 4., 3., 4.],\n
          [1., 2., 1., 2.],\n
          [3., 4., 3., 4.]]],\n
        [[[5., 6., 5., 6.],\n
          [7., 8., 7., 8.],\n
          [5., 6., 5., 6.],\n
          [7., 8., 7., 8.]]]])
    """
    x_ld = torch.flip(x_rd, dims=(-2,))
    x_d = torch.cat([x_ld, x_rd], dim=-2)
    x_u = torch.flip(x_d, dims=(-1,))
    x = torch.cat([x_u, x_d], dim=-1)
    return x

def over_op(alpha):
    bs, cs, ds, hs, ws = alpha.shape
    out = torch.cumprod(1. - alpha, dim=-3)
    return torch.cat([torch.ones((bs, cs, 1, hs, ws), dtype=out.dtype, device=out.device), out[:, :, :-1]], dim=-3)

def gray_to_rgb(x):
    return x.repeat(1, 3, 1, 1)


def linear_to_srgb(x, eps=1e-8):
    a = 0.055
    x = x.clamp(eps, 1.)
    return torch.where(x <= 0.0031308, 12.92 * x, (1. + a) * x ** (1. / 2.4) - a)


def srgb_to_linear(x, eps=1e-8):
    x = x.clamp(eps, 1.)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def to_bayer(x):
    mask = torch.zeros_like(x)
    # masking r
    mask[:, 0, ::2, ::2] = 1
    # masking b
    mask[:, 2, 1::2, 1::2] = 1
    # masking g
    mask[:, 1, 1::2, ::2] = 1
    mask[:, 1, ::2, 1::2] = 1
    y = x * mask
    bayer = y.sum(dim=1, keepdim=True)
    return bayer


def imresize(img, size):
    return F.interpolate(img, size=size)

def gray_to_rgb(x):
    return x.repeat(1, 3, 1, 1)

def crop_boundary(x, w):
    if w == 0:
        return x
    else:
        return x[..., w:-w, w:-w]
    
def cosine_similarity_matrices(A, B):
    A_flat = A.view(1, -1)  # Reshape A to a row tensor
    B_flat = B.view(1, -1)  # Reshape B to a row tensor
    cos_sim = F.cosine_similarity(A_flat, B_flat, dim=1)
    return cos_sim.item()

def normal_pdf(x, mean, std):
    # Compute the PDF of the normal distribution
    return (1 / (math.sqrt(2 * math.pi) * std)) * torch.exp(-((x - mean) ** 2) / (2 * std ** 2))

def create_inverse_normal_pdf2D(rows, cols,mean=0, std=0.1):
    # Create a grid of coordinates
    x_coords = torch.linspace(-1, 1, steps=cols)
    y_coords = torch.linspace(-1, 1, steps=rows)
    x_grid, y_grid = torch.meshgrid(x_coords, y_coords)
    
    # Compute the PDF values for each point
    pdf_values = normal_pdf(x_grid, mean, std) * normal_pdf(y_grid, mean, std)
    pdf_values -= pdf_values.min()
    pdf_values /= pdf_values.max()
    pdf_values = 1.0 - pdf_values
    return pdf_values