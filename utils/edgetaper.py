"""
Refer to
https://github.com/AndreiDavydov/Poisson_Denoiser/blob/master/pydl/nnLayers/functional/functional.py
under MIT Licence (copyright: Andrei Davydov)
"""
import torch

from utils import complex


def autocorrelation1d_symmetric(h):
    """Compute autocorrelation of a symmetric signal along the last dimension"""
    #Fhsq = complex.abs2(torch.rfft(h, 1))
    Fhsq = complex.abs2(torch.view_as_real(torch.fft.rfft(h)))
    #a = torch.irfft(torch.stack([Fhsq, torch.zeros_like(Fhsq)], dim=-1), 1, signal_sizes=(h.shape[-1],))
    a = torch.fft.irfft(torch.view_as_complex(torch.stack([Fhsq, torch.zeros_like(Fhsq)], dim=-1)), h.size(-1))
    return a / a.max() #B, C, _

def compute_weighting_for_tapering(h):
    # h:  B, C, H, W
    """Compute autocorrelation of a symmetric signal along the last two dimension"""
    h_proj0 = h.sum(dim=-2, keepdims=False) #B, C, W
    autocorr_h_proj0 = autocorrelation1d_symmetric(h_proj0).unsqueeze(-2) #B, C, 1,  W
    h_proj1 = h.sum(dim=-1, keepdims=False) #B, C, 1
    autocorr_h_proj1 = autocorrelation1d_symmetric(h_proj1).unsqueeze(-1) #B, C, 1, 1
    return (1 - autocorr_h_proj0) * (1 - autocorr_h_proj1) # B, C, H, W


def edgetaper3d(img, psf):
    """
    Edge-taper an image with a depth-dependent PSF

    Args:
        img: (B x C x H x W)
        psf: 3d rotationally-symmetric psf (B x C x D x H x W) (i.e. continuous at boundaries)

    Returns:
        Edge-tapered 3D image
    """
    assert (img.dim() == 4)
    assert (psf.dim() == 5)
    psf = psf.mean(dim=-3) # B, C, H, W
    alpha = compute_weighting_for_tapering(psf) # B, C, H, W
    # blurred_img = torch.irfft(
    #     complex.multiply(torch.rfft(img, 2), torch.rfft(psf, 2)), 2, signal_sizes=img.shape[-2:]
    # )
    blurred_img = torch.fft.irfft2(
         torch.view_as_complex(complex.multiply(torch.view_as_real(torch.fft.rfft2(img)), torch.view_as_real(torch.fft.rfft2(psf))))
                          , img.shape[-2:]
    ) # B, C, H, W
    return alpha * img + (1 - alpha) * blurred_img # B, C, H, W
