import torch
import torch.nn.functional as F

from utils import complex
from utils.edgetaper import edgetaper3d


def tikhonov_inverse_fast(Y, G, v=None, beta=0, gamma=1e-1, dataformats='SDHW'):
    """
    Compute Tikhonov-regularized inverse. This function solves
        argmin_x || y - sum_{k} G_k x_k ||^2 + beta sum_{k} || S x_k ||^2 + gamma || x - v ||^2
    x_k: k-th 2D slice of a 3D volume x
    S: 2D Laplacian operator

    gamma || x ||^2 seems to be required for numerical stability.

    original signal size is assumed to be even.

    Even though the element in the block-diagonal matrix in Woodbuery formula is described as a row vector in our
    equation, the implementation uses a vector for convenience. Be careful about its conjugate transpose.
    """
    if dataformats == 'DHW':
        Y = Y[None, None, None, ...]  # add batch, color and shot dimension
        G = G[None, None, None, ...]  # add batch, color and shot dimension
        if v is not None:
            v = v[None, None, None, ...]
    elif dataformats == 'SDHW':
        Y = Y[None, None, ...]  # add batch and color dimension
        G = G[None, None, ...]  # add batch and color dimension
        if v is not None:
            v = v[None, None, ...]
    elif dataformats == 'CSDHW':
        Y = Y[None, ...]  # add batch dimension
        G = G[None, ...]  # add batch dimension
        if v is not None:
            v = v[None, ...]
    elif dataformats == 'BCDHW':
        Y = Y.unsqueeze(2)  # add shot dimension
        G = G.unsqueeze(2)
        if v is not None:
            v = v.unsqueeze(2)
    elif dataformats == 'BCSDHW':
        pass
    else:
        raise NotImplementedError(f'This data format is not supported! [dataformats: {dataformats}]')

    device = Y.device
    dtype = Y.dtype
    num_colors, num_shots, depth, height, width = G.shape[1:6]
    batch_sz = Y.shape[0]

    Y_real = Y[..., 0].reshape([batch_sz, num_colors, num_shots, 1, -1]).transpose(2, 4)
    Y_imag = Y[..., 1].reshape([batch_sz, num_colors, num_shots, 1, -1]).transpose(2, 4)
    G_real = (G[..., 0]).reshape([1, num_colors, num_shots, depth, -1]).transpose(2, 4)
    G_imag = (G[..., 1]).reshape([1, num_colors, num_shots, depth, -1]).transpose(2, 4)
    Gc_real = G_real
    Gc_imag = -G_imag

    GcY_real = (Gc_real * Y_real - Gc_imag * Y_imag).sum(dim=-1, keepdims=True)
    GcY_imag = (Gc_imag * Y_real + Gc_real * Y_imag).sum(dim=-1, keepdims=True)

    # This part is still not covered in test!
    if v is not None:
        #V = gamma * torch.rfft(v, 2)
        V = gamma * torch.view_as_real(torch.fft.rfft2(v))
        V_real = (V[..., 0]).reshape([batch_sz, num_colors, 1, depth, -1]).transpose(2, 4)
        V_imag = (V[..., 1]).reshape([batch_sz, num_colors, 1, depth, -1]).transpose(2, 4)
        GcY_real += V_real
        GcY_imag += V_imag

    if not isinstance(gamma, torch.Tensor):
        reg = torch.tensor(gamma, device=device, dtype=dtype)
    else:
        reg = gamma

    Gc_real_t = Gc_real.transpose(3, 4)
    Gc_imag_t = Gc_imag.transpose(3, 4)
    # innerprod's imaginary part should be zero.
    # The conjugate transpose is implicitly reflected in the sign of complex multiplication.
    if num_shots == 1:
        innerprod = torch.matmul(Gc_real_t, G_real) - torch.matmul(Gc_imag_t, G_imag)
        outerprod_real = torch.matmul(G_real, Gc_real_t) - torch.matmul(G_imag, Gc_imag_t)
        outerprod_imag = torch.matmul(G_imag, Gc_real_t) + torch.matmul(G_real, Gc_imag_t)
        invM_real = 1. / reg * (
                torch.eye(depth, device=device, dtype=dtype) - outerprod_real / (reg + innerprod))
        invM_imag = -1. / reg * outerprod_imag / (reg + innerprod)
    else:
        eye_plus_inner = torch.eye(num_shots, device=device, dtype=dtype) + 1 / reg * (
                torch.matmul(Gc_real_t, G_real) - torch.matmul(Gc_imag_t, G_imag))
        eye_plus_inner_inv = torch.inverse(eye_plus_inner)
        inner_Gc_real = torch.matmul(eye_plus_inner_inv, Gc_real_t)
        inner_Gc_imag = torch.matmul(eye_plus_inner_inv, Gc_imag_t)
        prod_real = 1 / reg * (torch.matmul(G_real, inner_Gc_real) - torch.matmul(G_imag, inner_Gc_imag))
        prod_imag = 1 / reg * (torch.matmul(G_imag, inner_Gc_real) + torch.matmul(G_real, inner_Gc_imag))
        invM_real = 1 / reg * (torch.eye(depth, device=device, dtype=dtype).unsqueeze(0) - prod_real)
        invM_imag = - 1 / reg * prod_imag

    X_real = (torch.matmul(invM_real, GcY_real) - torch.matmul(invM_imag, GcY_imag))
    X_imag = (torch.matmul(invM_imag, GcY_real) + torch.matmul(invM_real, GcY_imag))
    X = torch.stack(
        [X_real.transpose(2, 3).reshape(batch_sz, num_colors, depth, height, width),
         X_imag.transpose(2, 3).reshape(batch_sz, num_colors, depth, height, width)],
        dim=-1)

    if dataformats == 'SDHW':
        X = X.reshape(depth, height, width, 2)
    elif dataformats == 'CSDHW':
        X = X.reshape(num_colors, depth, height, width, 2)
    elif dataformats == 'BCSDHW' or dataformats == 'BCDHW':
        X = X.reshape(batch_sz, num_colors, depth, height, width, 2)

    return X


def apply_tikhonov_inverse(captimg, psf, reg_tikhonov, apply_edgetaper=True):
    """

    Args:
        captimg: (B x C x H x W)
        psf: PSF lateral size should be equal to captimg. (1 x C x D x H x W)
        reg_tikhonov: float

    Returns:
        B x C x D x H x W

    """
    if apply_edgetaper:
        # Edge tapering
        captimg = edgetaper3d(captimg, psf)
    # Fpsf = torch.rfft(psf, 2)
    # Fcaptimgs = torch.rfft(captimg, 2)
    Fpsf = torch.view_as_real(torch.fft.rfft2(psf))
    Fcaptimgs = torch.view_as_real(torch.fft.rfft2(captimg))
    Fpsf = Fpsf.unsqueeze(2)  # add shot dim
    Fcaptimgs = Fcaptimgs.unsqueeze(2)  # add shot dim
    est_X = tikhonov_inverse_fast(Fcaptimgs, Fpsf, v=None, beta=0, gamma=reg_tikhonov,
                                  dataformats='BCSDHW')
    #est_volumes = torch.irfft(est_X, 2, signal_sizes=captimg.shape[-2:])
    est_volumes = torch.fft.irfft2(torch.view_as_complex(est_X), captimg.shape[-2:])
    return est_volumes
