import abc
from typing import List, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.special
import utils.helper
from utils import complex, interp
from utils.fft import fftshift
import cv2
class Camera(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, focal_depth, min_depth, max_depth, n_depths, image_size, mask_size,
                 focal_length, mask_diameter, camera_pixel_pitch, wavelengths, mask_upsample_factor=1, diffraction_efficiency=0.7,
                 full_size=100, debug = False, requires_grad = False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        assert min_depth > 1e-6, f'Minimum depth is too small. min_depth: {min_depth}'
        # Creates a one-dimensional tensor of size steps whose values are evenly spaced from start to end, inclusive. 
        scene_distances = utils.helper.ips_to_metric(torch.linspace(0, 1, steps=n_depths), min_depth, max_depth)
        self.debug = debug
        if (debug):
            print("max_depth - min_depth")
            print(max_depth - min_depth)
            print(scene_distances)
        self.n_depths = len(scene_distances)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.focal_depth = focal_depth
        self.focal_length = focal_length
        self.mask_diameter = mask_diameter

        # the distance between holes in the shadow mask. ex: distance between pĩxel
        self.mask_pitch = self.mask_diameter / mask_size
        self.camera_pixel_pitch = camera_pixel_pitch
        # number of pixel per diameter
        self.mask_size = mask_size

        self.image_size = self._normalize_image_size(image_size)
        self._register_wavlength(wavelengths)
        self.diffraction_efficiency = diffraction_efficiency

        # init_heightmap1d = torch.zeros(mask_size // 2 // mask_upsample_factor)  # 1D half size (radius)
        # self.heightmap1d_ = torch.nn.Parameter(init_heightmap1d, requires_grad=requires_grad)
        self.mask_upsample_factor = mask_upsample_factor

        # pin_hole camera
        # init_ampmask2d = torch.zeros(8,8)
        # self.ampmask2d = torch.nn.Parameter(init_ampmask2d, requires_grad=requires_grad)

        self.full_size = self._normalize_image_size(full_size)
        self.register_buffer('scene_distances', scene_distances)

        self.build_camera()

    @abc.abstractmethod
    def build_camera(self):
        pass

    def forward(self, img, depthmap, occlusion, is_training=torch.tensor(False)):
        """
        Args:
            img: B x C x H x W

        Returns:
            captured image: B x C x H x W
        """
        psf = self.psf_at_camera(size=img.shape[-2:], is_training=is_training).unsqueeze(0)  # add batch dimension
        # print(psf.shape) #torch.Size([b (1), c, d, w, h])

        psf = self.normalize_psf(psf)
        captimg, volume = self._capture_from_rgbd_with_psf_impl(img, depthmap, psf, occlusion)
        return captimg, volume, psf

    
    def forward_train(self, img, depthmap, occlusion):
        return self.forward(img, depthmap, occlusion, is_training=torch.tensor(True))

    def set_diffraction_efficiency(self, de: float):
        self.diffraction_efficiency = de
        self.build_camera()

    def psf_out_of_fov_energy(self, psf_size: int):
        """This can be run only after psf_at_camera is evaluated once."""
        device = self.device
        scene_distances = utils.helper.ips_to_metric(torch.linspace(0, 1, steps=self.n_depths, device=device),
                                        self.min_depth, self.max_depth)
        psf1d_diffracted = self.psf1d_full(scene_distances, torch.tensor(True))
        # Normalize PSF based on the cropped PSF
        psf1d_diffracted = psf1d_diffracted / self.diff_normalization_scaler.squeeze(-1)
        edge = psf_size / 2 * self.camera_pixel_pitch / (
                self.wavelengths.reshape(-1, 1, 1) * self.sensor_distance())
        psf1d_out_of_fov = psf1d_diffracted * (self.rho_grid_full.unsqueeze(1) > edge).float()
        return psf1d_out_of_fov.sum(), psf1d_out_of_fov.max()
    
    def psf1d_full(self, scene_distances, modulate_phase=torch.tensor(True)):
        return self.psf1d(self.H_full, scene_distances, modulate_phase=modulate_phase)

    def sensor_distance(self):
        return 1. / (1. / self.focal_length - 1. / self.focal_depth)
    
    def find_index(self, a, v):
        #The find_index function you've provided takes two tensors, a and v, and returns a tensor of indices where 
        #each index corresponds to the leftmost position in a where the corresponding element in v can be inserted
        #while maintaining the sorted order.
        a = a.squeeze(1).cpu().numpy()
        v = v.cpu().numpy()
        index = np.stack([np.searchsorted(a[i, :], v[i], side='left') - 1 for i in range(a.shape[0])], axis=0)
        return torch.from_numpy(index)
    
    def pointsource_inputfield1d(self):
        pass

    def normalize_psf(self, psfimg):
        # Scale the psf
        # As the incoming light doesn't change, we compute the PSF energy without the phase modulation
        # and use it to normalize PSF with phase modulation.
        return psfimg / psfimg.sum(dim=(-2, -1), keepdims=True)
        # n_wl X D x H x W  / n_wl x D x 1 x 1 -> n_wl X D x H x W
        # each pixel divide by the sum of whole image pixel
    
    def psf1d(self, modulate_phase=torch.tensor(True)):
        pass

    def _psf_at_camera_impl(self):
        # As this quadruple will be copied to the other three, rho = 0 is avoided.
        pass
        # psf1d: n_wl X D X n_rho
        # rho_grid: n_wl X n_rho
        # rho_sampling: n_wl X (n_rho - 2) X (n_rho - 2)
        # ind: n_wl X (n_rho - 2) X (n_rho - 2)

    def psf_at_camera(self, size=None, modulate_phase=torch.tensor(True), **kwargs):
        #
        psf = self._psf_at_camera_impl(
            self.H, self.rho_grid, self.rho_sampling, self.ind, self.image_size, modulate_phase)
        #pad 0 to the psf
        if size is not None:
            pad_h = (size[0] - self.image_size[0]) // 2
            pad_w = (size[1] - self.image_size[1]) // 2
            psf = F.pad(psf, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
        return fftshift(psf, dims=(-1, -2))
    
    def _capture_impl(self, volume, layered_depth, psf, occlusion, eps=1e-3):
        scale = volume.max()
        volume = volume / scale   # (B, 3, D, H, W)
        Fpsf = torch.view_as_real(torch.fft.rfft2(psf))
        if(self.debug):
            print("volume: ")
            print(volume.shape)

        if occlusion:
            #Fvolume = torch.rfft(volume, 2)
            Fvolume = torch.view_as_real(torch.fft.rfft2(volume))
            #Flayered_depth = torch.rfft(layered_depth, 2)
            Flayered_depth = torch.view_as_real(torch.fft.rfft2(layered_depth))
            # blurred_alpha_rgb = torch.irfft(
            #     complex.multiply(Flayered_depth, Fpsf), 2, signal_sizes=volume.shape[-2:])
            # blurred_volume = torch.irfft(
            #     complex.multiply(Fvolume, Fpsf), 2, signal_sizes=volume.shape[-2:])
            blurred_alpha_rgb = torch.fft.irfft2(torch.view_as_complex(complex.multiply(Flayered_depth, Fpsf)), volume.shape[-2:])
            blurred_volume = torch.fft.irfft2(torch.view_as_complex(complex.multiply(Fvolume, Fpsf)), volume.shape[-2:])
            # Normalize the blurred intensity
            cumsum_alpha = torch.flip(torch.cumsum(torch.flip(layered_depth, dims=(-3,)), dim=-3), dims=(-3,))
            #Fcumsum_alpha = torch.rfft(cumsum_alpha, 2)
            Fcumsum_alpha = torch.view_as_real(torch.fft.rfft2(cumsum_alpha))
            # blurred_cumsum_alpha = torch.irfft(
            #     complex.multiply(Fcumsum_alpha, Fpsf), 2, signal_sizes=volume.shape[-2:])
            blurred_cumsum_alpha = torch.fft.irfft2(torch.view_as_complex(complex.multiply(Fcumsum_alpha, Fpsf)), volume.shape[-2:])
            blurred_volume = blurred_volume / (blurred_cumsum_alpha + eps)
            blurred_alpha_rgb = blurred_alpha_rgb / (blurred_cumsum_alpha + eps)

            over_alpha = utils.helper.over_op(blurred_alpha_rgb)
            captimg = torch.sum(over_alpha * blurred_volume, dim=-3)
        else:
            #Fvolume = torch.rfft(volume, 2)
            Fvolume = torch.view_as_real(torch.fft.rfft2(volume))
            Fcaptimg = complex.multiply(Fvolume, Fpsf).sum(dim=2)
            #captimg = torch.irfft(Fcaptimg, 2, signal_sizes=volume.shape[-2:])
            captimg = torch.fft.irfft2(torch.view_as_complex(Fcaptimg), volume.shape[-2:])

        captimg = scale * captimg
        volume = scale * volume
        return captimg, volume
    
    def _capture_from_rgbd_with_psf_impl(self, img, depthmap, psf, occlusion):
        layered_depth = utils.helper.depthmap_to_layereddepth(depthmap, self.n_depths, binary=True)
        volume = layered_depth * img[:, :, None, ...] #return the (B, 3, D, H, W) RGB image at each depth level
        return self._capture_impl(volume, layered_depth, psf, occlusion)
    
    def capture_from_rgbd_with_psf(self, img, depthmap, psf, occlusion):
        return self._capture_from_rgbd_with_psf_impl(img, depthmap, psf, occlusion)[0]
    
    def capture_from_rgbd(self, img, depthmap, occlusion):
        psf = self.psf_at_camera(size=img.shape[-2:]).unsqueeze(0)  # add batch dimension
        psf = self.normalize_psf(psf)  # n_wl X D x H x W
        return self.capture_from_rgbd_with_psf(img, depthmap, psf, occlusion)
    
    def _normalize_image_size(self, image_size):
        if isinstance(image_size, int):
            image_size = [image_size, image_size]
        elif isinstance(image_size, list):
            if image_size[0] % 2 == 1 or image_size[1] % 2 == 1:
                raise ValueError('Image size has to be even.')
        else:
            raise ValueError('image_size has to be int or list of int.')
        return image_size
    
    def _register_wavlength(self, wavelengths):
        if isinstance(wavelengths, list):
            wavelengths = torch.tensor(wavelengths)  # in [meter]
        elif isinstance(wavelengths, float):
            wavelengths = torch.tensor([wavelengths])
        else:
            raise ValueError('wavelengths has to be a float or a list of floats.')

        if len(wavelengths) % 3 != 0:
            raise ValueError('the number of wavelengths has to be a multiple of 3.')

        self.n_wl = len(wavelengths)
        #TOCHANGE
        wavelengths = wavelengths.to(self.device)
        #
        if not hasattr(self, 'wavelengths'):
            self.register_buffer('wavelengths', wavelengths)
        else:
            self.wavelengths = wavelengths.to(self.wavelengths.device)

    @abc.abstractmethod
    def heightmap(self):
        pass
    
class RotationallySymmetricCamera(Camera):
    def __init__(self, focal_depth, min_depth, max_depth, n_depths, image_size, mask_size,
                 focal_length, mask_diameter, camera_pixel_pitch, wavelengths, mask_upsample_factor=1, diffraction_efficiency=0.7,
                 full_size=100, debug = False, requires_grad = False):
        super().__init__(focal_depth, min_depth, max_depth, n_depths, image_size, mask_size,
                 focal_length, mask_diameter, camera_pixel_pitch, wavelengths, mask_upsample_factor, diffraction_efficiency,
                 full_size, debug, requires_grad)
        init_heightmap1d = torch.zeros(mask_size // 2 // mask_upsample_factor)  # 1D half size (radius)
        #TOCHANGE
        #self.heightmap1d_ = torch.nn.Parameter(init_heightmap1d, requires_grad=requires_grad)

    def build_camera(self):
        pass
        # H, rho_grid, rho_sampling = self.precompute_H(self.image_size)
        # ind = self.find_index(rho_grid, rho_sampling)

        # H_full, rho_grid_full, rho_sampling_full = self.precompute_H(self.full_size)
        # ind_full = self.find_index(rho_grid_full, rho_sampling_full)

        # assert (rho_grid.max(dim=-1)[0] >= rho_sampling.reshape(self.n_wl, -1).max(dim=-1)[0]).all(), \
        #     'Grid (max): {}, Sampling (max): {}'.format(
        #         rho_grid.max(dim=-1)[0], rho_sampling.reshape(self.n_wl, -1).max(dim=-1)[0])
        # assert (rho_grid.min(dim=-1)[0] <= rho_sampling.reshape(self.n_wl, -1).min(dim=-1)[0]).all(), \
        #     'Grid (min): {}, Sampling (min): {}'.format(
        #         rho_grid.min(dim=-1)[0], rho_sampling.reshape(self.n_wl, -1).min(dim=-1)[0])
        # self.register_buffer('H', H)
        # self.register_buffer('rho_grid', rho_grid)
        # self.register_buffer('rho_sampling', rho_sampling)
        # self.register_buffer('ind', ind)
        # self.register_buffer('H_full', H_full)
        # self.register_buffer('rho_grid_full', rho_grid_full)
        # # These two parameters are not used for training.
        # self.rho_sampling_full = rho_sampling_full
        # self.ind_full = ind_full

    def heightmap1d(self):
        #F.interpolate(..., mode='nearest'): Applies nearest-neighbor interpolation to the 3D tensor obtained in the previous step. 
        # The F.interpolate function is often used for upsampling or downsampling tensors. In this case,
        #  it seems like it's being used to change the size of the tensor.
        #.reshape(-1): flatten tensor to 1D
        return F.interpolate(self.heightmap1d_.reshape(1, 1, -1),
                             scale_factor=self.mask_upsample_factor, mode='nearest').reshape(-1)
        # this made currently did nothing

    def heightmap(self):
        heightmap1d = torch.cat([self.heightmap1d().cpu(), torch.zeros((self.mask_size // 2))], dim=0)
        heightmap1d = heightmap1d.reshape(1, 1, -1)
        r_grid = torch.arange(0, self.mask_size, dtype=torch.double)
        y_coord = torch.arange(0, self.mask_size // 2, dtype=torch.double).reshape(-1, 1) + 0.5
        x_coord = torch.arange(0, self.mask_size // 2, dtype=torch.double).reshape(1, -1) + 0.5
        r_coord = torch.sqrt(y_coord ** 2 + x_coord ** 2).unsqueeze(0)
        r_grid = r_grid.reshape(1, -1)
        ind = self.find_index(r_grid, r_coord)
        heightmap11 = interp.interp(r_grid, heightmap1d, r_coord, ind).float()
        heightmap = utils.helper.copy_quadruple(heightmap11).squeeze()
        return heightmap
    
    def pointsource_inputfield1d(self, scene_distances):
        #device = scene_distances.device
        #TOCHANGE
        scene_distances = scene_distances.to(self.device)
        r = self.mask_pitch * torch.linspace(1, self.mask_size / 2, self.mask_size // 2, device=self.device).double()
        #
        # compute pupil function
        wavelengths = self.wavelengths.reshape(-1, 1, 1).double()
        scene_distances = scene_distances.reshape(1, -1, 1).double()  # 1 x D x 1
        r = r.reshape(1, 1, -1)
        wave_number = 2 * math.pi / wavelengths

        radius = torch.sqrt(scene_distances ** 2 + r ** 2)  # 1 x D x n_r

        # ignore 1/j term (constant phase)
        amplitude = scene_distances / wavelengths / radius ** 2  # n_wl x D x n_r
        amplitude /= amplitude.max()
        # zero phase at center
        phase = wave_number * (radius - scene_distances)  # n_wl x D x n_r
        if not math.isinf(self.focal_depth):
            #TOCHANGE
            focal_depth = torch.tensor(self.focal_depth, device=self.device).reshape(1, 1, 1).double()  # 1 x 1 x 1
            f_radius = torch.sqrt(focal_depth ** 2 + r ** 2)  # 1 x 1 x n_r
            phase -= wave_number * (f_radius - focal_depth)  # subtract focal_depth to roughly remove a piston
        return amplitude, phase # n_wl X D x r
    
    def psf1d(self, H, scene_distances, modulate_phase=torch.tensor(True)):
        """Perform all computations in double for better precision. Float computation fails."""
        prop_amplitude, prop_phase = self.pointsource_inputfield1d(scene_distances) # n_wl X D x r

        H = H.unsqueeze(1)  # n_wl x 1 x n_r x n_rho
        wavelengths = self.wavelengths.reshape(-1, 1, 1).double()
        if modulate_phase:
            phase_delays =  utils.helper.heightmap_to_phase(self.heightmap1d().reshape(1, -1),  # add wavelength dim
                                              wavelengths,
                                               utils.helper.refractive_index(wavelengths))
            phase = phase_delays + prop_phase  # n_wl X D x n_r
        else:
            phase = prop_phase

        # broadcast the matrix-vector multiplication
        phase = phase.unsqueeze(2)  # n_wl X D X 1 x n_r
        amplitude = prop_amplitude.unsqueeze(2)  # n_wl X D X 1 x n_r
        real = torch.matmul(amplitude * torch.cos(phase), H).squeeze(-2)
        imag = torch.matmul(amplitude * torch.sin(phase), H).squeeze(-2)

        return (2 * math.pi / wavelengths / self.sensor_distance()) ** 2 * (real ** 2 + imag ** 2)  # n_wl X D X n_rho

    def psf1d_full(self, scene_distances, modulate_phase=torch.tensor(True)):
        #return self.psf1d(self.H_full, scene_distances, modulate_phase=modulate_phase)
        return self.psf1d_(scene_distances, modulate_phase=modulate_phase)
    
    def _psf_at_camera_impl(self, H, rho_grid, rho_sampling, ind, size, scene_distances, modulate_phase):
        # As this quadruple will be copied to the other three, rho = 0 is avoided.
        psf1d = self.psf1d(H, scene_distances, modulate_phase)
        psf_rd = F.relu(interp.interp(rho_grid, psf1d, rho_sampling, ind).float())
        psf_rd = psf_rd.reshape(self.n_wl, self.n_depths, size[0] // 2, size[1] // 2)
        return utils.helper.copy_quadruple(psf_rd) # wl x depth x size x size

    def psf_at_camera(self, size=None, modulate_phase=torch.tensor(True), is_training=torch.tensor(False)):
        device = self.device
        if is_training:
            # scene_distances = utils.helper.ips_to_metric(
            #     torch.linspace(0, 1, steps=self.n_depths, device=device) +
            #     1 / self.n_depths * (torch.rand(self.n_depths, device=device) - 0.5),
            #     self.min_depth, self.max_depth)
            # scene_distances[-1] += torch.rand(1, device=device)[0] * (100.0 - self.max_depth)
            scene_distances = self.scene_distances
        else:
            scene_distances = utils.helper.ips_to_metric(torch.linspace(0, 1, steps=self.n_depths, device=device),
                                            self.min_depth, self.max_depth)
            # scene_distances = self.scene_distances

        # diffracted_psf = self._psf_at_camera_impl(
        #     scene_distances, modulate_phase)
        diffracted_psf = self.psf_at_camera
        undiffracted_psf, _ = self._psf_at_camera_impl(
           scene_distances, torch.tensor(False))
        
        # print(undiffracted_psf.shape) #torch.Size([c, d, w, h])
        # print(undiffracted_psf.dtype) #torch.float32
        
        # Keep the normalization factor for penalty computation
        self.diff_normalization_scaler = diffracted_psf.sum(dim=(-1, -2), keepdim=True)
        self.undiff_normalization_scaler = undiffracted_psf.sum(dim=(-1, -2), keepdim=True)

        diffracted_psf = diffracted_psf / self.diff_normalization_scaler
        undiffracted_psf = undiffracted_psf / self.undiff_normalization_scaler

        psf = diffracted_psf 

        # In training, randomly pixel-shifts the PSF around green channel.
        if is_training:
            if (self.debug):
                print("in training")
            max_shift = 2
            r_shift = tuple(np.random.randint(low=-max_shift, high=max_shift, size=2))
            b_shift = tuple(np.random.randint(low=-max_shift, high=max_shift, size=2))
            psf_r = torch.roll(psf[0], shifts=r_shift, dims=(-1, -2))
            psf_g = psf[1]
            psf_b = torch.roll(psf[2], shifts=b_shift, dims=(-1, -2))
            psf = torch.stack([psf_r, psf_g, psf_b], dim=0)

        if torch.tensor(size is not None):
            pad_h = (size[0] - self.image_size[0]) // 2
            pad_w = (size[1] - self.image_size[1]) // 2
            psf = F.pad(psf, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
        return fftshift(psf, dims=(-1, -2))

    def psf_out_of_fov_energy(self, psf_size: int):
        """This can be run only after psf_at_camera is evaluated once."""
        #device = self.H.device
        device = self.device
        scene_distances = utils.helper.ips_to_metric(torch.linspace(0, 1, steps=self.n_depths, device=device),
                                        self.min_depth, self.max_depth)
        # psf1d_diffracted = self.psf1d_full(scene_distances, torch.tensor(True))
        # psf1d_diffracted = self._psf_at_camera_impl_full(scene_distances, torch.tensor(True))
        psf1d_diffracted = self.psf_at_camera_full
        # Normalize PSF based on the cropped PSF

        #dump
        if (self.debug):
            print("psf1d_diffracted")
            print(psf1d_diffracted.shape)
            print(" self.diff_normalization_scaler.squeeze(-1)")
            print( self.diff_normalization_scaler.shape)
            print( psf1d_diffracted / self.diff_normalization_scaler)
        #psf1d_diffracted = psf1d_diffracted / self.diff_normalization_scaler.squeeze(-1)
        psf1d_diffracted = psf1d_diffracted / self.diff_normalization_scaler
        # edge = psf_size / 2 * self.camera_pixel_pitch / (
        #         self.wavelengths.reshape(-1, 1, 1) * self.sensor_distance())
        # psf1d_out_of_fov = psf1d_diffracted * (self.rho_grid_full.unsqueeze(1) > edge).float()

        #create a mask 
        mask = torch.ones(3, self.n_depths, self.full_size[0], self.full_size[1]).to(self.device)
        uncropped_size = mask.shape
        uncropped_height = uncropped_size[-1]
        uncropped_width = uncropped_size[-2]
        left = (uncropped_width - psf_size) // 2
        top = (uncropped_height - psf_size) // 2
        right = (uncropped_width + psf_size) // 2
        bottom = (uncropped_height + psf_size) // 2
        mask[..., left:right, top:bottom] = 0
        psf1d_out_of_fov = psf1d_diffracted * mask.float()
        return psf1d_out_of_fov.sum(), psf1d_out_of_fov.max()    

class AsymmetricMaskRotationallySymmetricCamera(RotationallySymmetricCamera):
    def __init__(self, focal_depth, min_depth, max_depth, n_depths, image_size, mask_size,
                 focal_length, mask_diameter, camera_pixel_pitch, wavelengths, mask_upsample_factor=1, diffraction_efficiency=0.7,
                 full_size=100, debug = False, requires_grad = False):
        super().__init__(focal_depth, min_depth, max_depth, n_depths, image_size, mask_size,
                 focal_length, mask_diameter, camera_pixel_pitch, wavelengths, mask_upsample_factor, diffraction_efficiency,
                 full_size, debug, requires_grad)
        
        # Amplitude Square-pinhole-mask initialization
        # init_ampmask2d = torch.zeros(256,256, dtype=torch.float32)
        # init_ampmask2d[127,127] = float(1.0)

        # Amplitude Cicular-fully-openned-mask initialization
        init_ampmask2d = self.copy_quadruple(self.circular_mask(64))
        self.ampmask2d = init_ampmask2d
        #self.ampmask2d = torch.nn.Parameter(init_ampmask2d, requires_grad=requires_grad)

        # Zeros-heightmap initialization
        #self.heightmap1d_ = torch.zeros(mask_size // 2 // mask_upsample_factor).to(self.device)

        # Test-hightmap
        # self.heightmap1d_ = self.test_heightmap_diag()
        init_heightmap1d = torch.zeros(mask_size // 2)  # 1D half size (radius diag)
        self.heightmap1d_ = torch.nn.Parameter(init_heightmap1d, requires_grad=requires_grad)

    def build_camera(self):
        self.rho_grid, self.rho_sampling = self.pre_sampling()
        self.rho_index = self.find_index(self.rho_grid, self.rho_sampling)
        self.ind = self.rho_index
        #self.psf_at_camera, self.psf_at_camera_full = self._psf_at_camera_impl(scene_distances, modulate_phase)
        # self.rho_grid_full, self.rho_sampling_full = self.pre_sampling_fullsize()
        # self.ind_full = self.find_index(self.rho_grid_full, self.rho_sampling_full)
        

    # def find_index(self, a, v):
    #     '''
    #     The find_index function takes two tensors, a and v, and returns a tensor of indices where 
    #     each index corresponds to the leftmost position in a where the corresponding element in v can be inserted
    #     while maintaining the sorted order.\n
    #     It should return a tensor with same shape with v, and have value is in the range of index of elements in a.
    #     '''
    #     a = a.cpu().numpy()
    #     v = v.cpu().numpy()
    #     index = np.searchsorted(a[:], v, side='left') - 1 
    #     return torch.from_numpy(index)
    
    def find_index(self, a, v):
        '''
        The find_index function takes two tensors, a and v, and returns a tensor of indices where 
        each index corresponds to the leftmost position in a where the corresponding element in v can be inserted
        while maintaining the sorted order.\n
        It should return a tensor with same shape with v, and have value is in the range of index of elements in a.
        '''
        a = a.cpu().numpy()
        v = v.cpu().numpy()
        index = np.searchsorted(a[:], v, side='right')  
        index = torch.from_numpy(index)
        return torch.where(index > (self.mask_size // 2 + 1), torch.tensor(self.mask_size // 2 + 1), index)
  
    def heightmap_(self):
        ''' ??? \n
        Rotationally transforming a 1D heightmap (radius) into fully 2D heightmap (circle).\n
        This transformation is symmetric.\n
        '''
        #heightmap1d = torch.cat([self.heightmap1d().to(self.device), torch.zeros((self.full_size[0] // 2)).to(self.device)], dim=0)
        heightmap1d = self.heightmap1d_
        heightmap1d = heightmap1d.reshape(1, 1, -1)
        r_grid = torch.arange(0, self.mask_size, dtype=torch.double)
        y_coord = torch.arange(0, self.mask_size // 2, dtype=torch.double).reshape(-1, 1) + 0.5
        x_coord = torch.arange(0, self.mask_size // 2, dtype=torch.double).reshape(1, -1) + 0.5
        r_coord = torch.sqrt(y_coord ** 2 + x_coord ** 2).unsqueeze(0)
        r_grid = r_grid.reshape(1, -1)
        ind = self.find_index(r_grid, r_coord)
        heightmap11 = interp.interp(r_grid, heightmap1d, r_coord, ind).float()
        heightmap = utils.helper.copy_quadruple(heightmap11).squeeze()
        return heightmap
        

    def _mask_upscale(self):
        '''
        Parameterizing amplitude mask into low-dim n x n resolution.\n
        This function is intend to scale that into the mask resolution.\n
        '''
        # make sure that we always get Amp-mask laid into [0, 1]
        tanh_mask2D = torch.tanh(self.ampmask2d)
        new_mask = F.interpolate(tanh_mask2D.unsqueeze(0).unsqueeze(0), size=(self.mask_size, self.mask_size), mode='nearest').squeeze()
        return new_mask
    def radius_sampling(self):
        coord_y = self.mask_pitch * torch.arange(1, self.mask_size // 2 + 1).reshape(-1, 1)
        coord_x = self.mask_pitch * torch.arange(1, self.mask_size // 2 + 1).reshape(1, -1)
        coord_y = coord_y.double()
        coord_x = coord_x.double()
        # (mask_size / 2) x (mask_size / 2)
        r_sampling = torch.sqrt(coord_y ** 2 + coord_x ** 2)
        r_sampling =  r_sampling.to(self.device)
        
        r_grid = math.sqrt(2) * self.mask_pitch * (
                torch.arange(1, self.mask_size // 2 + 1, dtype=torch.double) + 0.5)
        #TOCHANGE
        r_grid = r_grid.to(self.device)
        return r_grid.unsqueeze(0), r_sampling.unsqueeze(0)
    
    def _rho_sampling(self):
        coord_y = self.camera_pixel_pitch * (torch.arange(1, self.image_size[0] // 2 + 1) - 0.5).reshape(-1, 1)
        coord_x = self.camera_pixel_pitch * (torch.arange(1, self.image_size[1] // 2 + 1) - 0.5).reshape(1, -1)
        coord_y = coord_y.double()
        coord_x = coord_x.double()
        # (mask_size / 2) x (mask_size / 2)
        r_sampling = torch.sqrt(coord_y ** 2 + coord_x ** 2)
        r_sampling =  r_sampling.to(self.device)
        
        r_grid = math.sqrt(2) * self.camera_pixel_pitch * (
                torch.arange(1, max(self.image_size) // 2 + 1, dtype=torch.double) - 0.5)
        #TOCHANGE
        r_grid = r_grid.to(self.device)
        return r_grid, r_sampling
    
    # def pre_sampling(self):
    #     r_grid = self.mask_pitch * (torch.arange(0, self.mask_size // 2 + 2 ))
    #     coord_y = self.mask_pitch * (torch.arange(0, self.mask_size // 2 )).reshape(-1, 1)
    #     coord_x = self.mask_pitch * (torch.arange(0, self.mask_size // 2 )).reshape(1, -1)
    #     coord_y = coord_y.double()
    #     coord_x = coord_x.double()
    #     # (mask_size / 2) x (mask_size / 2)
    #     r_sampling = torch.sqrt(coord_y ** 2 + coord_x ** 2)
    #     r_sampling =  r_sampling.to(self.device)
        
    #     #TOCHANGE
    #     r_grid = r_grid.to(self.device)
    #     return r_grid, r_sampling
    
    def pre_sampling(self):
        coord_y = self.mask_pitch * (torch.arange(0, self.mask_size // 2)).reshape(-1, 1)
        coord_x = self.mask_pitch * (torch.arange(0, self.mask_size // 2)).reshape(1, -1)
        coord_y = coord_y.double()
        coord_x = coord_x.double()
        # (mask_size / 2) x (mask_size / 2)
        r_sampling = torch.sqrt(coord_y ** 2 + coord_x ** 2)
        r_sampling =  r_sampling.to(self.device)
        
        r_grid = math.sqrt(2) * self.mask_pitch * (torch.arange(0, self.mask_size // 2 + 1, dtype=torch.double))
        #TOCHANGE
        r_grid = r_grid.to(self.device)
        return r_grid, r_sampling
    def pre_sampling_fullsize(self):
        coord_y = self.mask_pitch * (torch.arange(0, self.full_size[0] // 2)).reshape(-1, 1)
        coord_x = self.mask_pitch * (torch.arange(0, self.full_size[0] // 2)).reshape(1, -1)
        coord_y = coord_y.double()
        coord_x = coord_x.double()
        # (mask_size / 2) x (mask_size / 2)
        r_sampling = torch.sqrt(coord_y ** 2 + coord_x ** 2)
        r_sampling =  r_sampling.to(self.device)
        
        r_grid = math.sqrt(2) * self.mask_pitch * (torch.arange(0, self.full_size[0] // 2 + 1, dtype=torch.double))
        #TOCHANGE
        r_grid = r_grid.to(self.device)
        return r_grid, r_sampling
    def precompute_H(self, image_size):
        """
        This is assuming that the defocus phase doesn't change much in one pixel.
        Therefore, the mask_size has to be sufficiently large.
        """
        sensor_distance = 1
        wavelengths = torch.ones(3)
        # As this quadruple will be copied to the other three, zero is avoided.
        coord_y = self.camera_pixel_pitch * torch.arange(1, image_size // 2 + 1).reshape(-1, 1)
        coord_x = self.camera_pixel_pitch * torch.arange(1, image_size // 2 + 1).reshape(1, -1)
        coord_y = coord_y.double()
        coord_x = coord_x.double()
        rho_sampling = torch.sqrt(coord_y ** 2 + coord_x ** 2)

        # Avoiding zero as the numerical derivative is not good at zero
        # sqrt(2) is for finding the diagonal of FoV.
        rho_grid = math.sqrt(2) * self.camera_pixel_pitch * (
                torch.arange(-1,    image_size // 2 + 1, dtype=torch.double) + 0.5)

        # n_wl x 1 x n_rho_grid
        rho_grid = rho_grid.reshape(1, 1, -1) / (wavelengths.reshape(-1, 1, 1) * sensor_distance)
        # n_wl X (image_size[0]//2 + 1) X (image_size[1]//2 + 1)
        rho_sampling = rho_sampling.unsqueeze(0) / (wavelengths.reshape(-1, 1, 1) * sensor_distance)

        return  rho_grid.squeeze(1), rho_sampling

    def mask2sensor_scale(self, value):
        '''
        Rescaling mask res to fit sensor res and make sure they have the same length (meter).
        '''
        sensor_coord_ = torch.arange(1, self.image_size[0] // 2 + 1).to(self.device)
        # Transform a point on the sensor-res coordinate to mask-res coordinate
        r_coord = sensor_coord_ * self.camera_pixel_pitch / self.mask_pitch
        diff = r_coord - torch.floor(r_coord)
        # Assume that our sample lie on x.5 (x is unsigned int)
        # Calculate the blending factor between two consecutive sample point
        alpha = torch.where(diff <= 0.5, diff + 0.5, diff - 0.5)
        # Transform .5 coordinate to integer coordinate
        index = torch.floor(r_coord - 0.5).to(torch.int)
        # Mask out the out of bound index
        mask1 = index < 0 
        mask2 = index == (value.shape[-1] - 1)
        index[mask1] = 0

        desired_size = torch.max(index)
        # Calculate the padding size for the 1D tensor
        padding = max(0, desired_size - value.shape[-1] + 2)
        pad_value = F.pad(value, (0, padding))

        result = (1 - alpha) * pad_value[:, :, index] + alpha * pad_value[:, :, index + 1]
        # Mirror-coppying the out of bound index
        result[:, :, mask1] = value[:, :, 0].unsqueeze(-1)
        result[:, :, mask2] = value[:, :, -1].unsqueeze(-1)
        return result
    
    def defocus_factor(self, scene_distances):
        x = self.mask_pitch * torch.arange( - (self.mask_size // 2), self.mask_size // 2).double().reshape(-1, 1)
        y = self.mask_pitch * torch.arange( - (self.mask_size // 2), self.mask_size // 2).double().reshape(1, -1)
        r_squared = x ** 2 + y ** 2
        r_squared = r_squared.unsqueeze(0).to(self.device)
        z = scene_distances
        term = torch.sqrt(z.reshape(-1, 1, 1) ** 2 + r_squared)
        k = (2 * math.pi) / self.wavelengths.reshape(-1, 1, 1, 1).double()
        result = k * term.unsqueeze(0)
        # print('result')
        # print(result.shape)
        return result    
  
    def psf_at_camera(self, size=None, modulate_phase=torch.tensor(True), is_training=torch.tensor(False)):
        device = self.device
        if is_training:
            scene_distances = utils.helper.ips_to_metric(
                torch.linspace(0, 1, steps=self.n_depths, device=device) +
                1 / self.n_depths * (torch.rand(self.n_depths, device=device) - 0.5),
                self.min_depth, self.max_depth)
            scene_distances[-1] += torch.rand(1, device=device)[0] * (100.0 - self.max_depth)
        else:
            scene_distances = utils.helper.ips_to_metric(torch.linspace(0, 1, steps=self.n_depths, device=device),
                                            self.min_depth, self.max_depth)

        diffracted_psf = self._psf_at_camera_impl(
            scene_distances, modulate_phase)
        undiffracted_psf = self._psf_at_camera_impl(
            scene_distances, torch.tensor(False))

        # Keep the normalization factor for penalty computation
        self.diff_normalization_scaler = diffracted_psf.sum(dim=(-1, -2), keepdim=True)
        self.undiff_normalization_scaler = undiffracted_psf.sum(dim=(-1, -2), keepdim=True)

        diffracted_psf = diffracted_psf / self.diff_normalization_scaler
        undiffracted_psf = undiffracted_psf / self.undiff_normalization_scaler

        psf = self.diffraction_efficiency * diffracted_psf + (1 - self.diffraction_efficiency) * undiffracted_psf

        # In training, randomly pixel-shifts the PSF around green channel.
        if is_training:
            max_shift = 2
            r_shift = tuple(np.random.randint(low=-max_shift, high=max_shift, size=2))
            b_shift = tuple(np.random.randint(low=-max_shift, high=max_shift, size=2))
            psf_r = torch.roll(psf[0], shifts=r_shift, dims=(-1, -2))
            psf_g = psf[1]
            psf_b = torch.roll(psf[2], shifts=b_shift, dims=(-1, -2))
            psf = torch.stack([psf_r, psf_g, psf_b], dim=0)

        if torch.tensor(size is not None):
            pad_h = (size[0] - self.image_size[0]) // 2
            pad_w = (size[1] - self.image_size[1]) // 2
            psf = F.pad(psf, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
        return fftshift(psf, dims=(-1, -2))

    def psf1d(self, scene_distances, modulate_phase=torch.tensor(True)):
        if(self.debug):
            print("scene_distance: ")
            print(scene_distances)
        """
        return the PSF2D\n
        note: Perform all computations in double for better precision. Float computation fails."""
        wavelengths = self.wavelengths.reshape(-1, 1, 1).double()

        phasedelay_heightmap2d = utils.helper.heightmap_to_phase(self.heightmap1d_.unsqueeze(0),  # add wavelength dim
                                              wavelengths,
                                              1.5)
        phasedelay_heightmap2d = utils.helper.copy_quadruple(interp.linterp2(self.rho_grid, (phasedelay_heightmap2d), self.rho_sampling, self.rho_index))
        self.dump_heightmap2d = phasedelay_heightmap2d
        
        defocus_phase = self.defocus_factor(self.scene_distances)
        self.dump_defocus_phase = defocus_phase
        defocus_amplitude = amplitude = torch.ones_like(defocus_phase).to(self.device)
        self.dump_defocus_amplitude = defocus_amplitude
        #
        mask2D = self._mask_upscale().to(self.device) # x X y  
        phase = defocus_phase + phasedelay_heightmap2d

        #dump things
        self.dump_init_phase = phase # n_wl x D x X x Y
        self.dump_init_amplitude = amplitude # n_wl x D x X x Y
        self.dump_mask2D = mask2D
        #<--------------------------

        amplitude = amplitude * mask2D # n_wl x D x X x Y
        self.dump_amplitude = amplitude
        #in_camera_phase = self.to_sensor_phase(700)
        in_camera_phase = self.to_sensor_phase(400)
        new_size = amplitude.shape[-1] + in_camera_phase.shape[-1] - 1

        phase_complex = torch.complex(amplitude * torch.cos(phase), amplitude * torch.sin(phase))
        phase_complex = self.__padding2size(phase_complex, new_size)
        
        in_camera_phase_complex = torch.complex(torch.cos(in_camera_phase), torch.sin(in_camera_phase)).unsqueeze(0) # x,y
        in_camera_phase_complex = in_camera_phase_complex / self.wavelengths.reshape(-1, 1, 1)  # n_wl , x, y
        self.dump_h_prepad = in_camera_phase_complex        
        in_camera_phase_complex = self.__padding2size(in_camera_phase_complex, new_size) 
        in_camera_phase_complex = in_camera_phase_complex.squeeze(0).unsqueeze(1)

        #-------------------------->    
        #TODUMP
        self.dump_phase_complex = phase_complex
        self.dump_h = in_camera_phase_complex
        #<--------------------------
        #TODUMP
        self.dump_phase_complex_fft = torch.fft.fft2(phase_complex)
        self.dump_h_complex_fft = torch.fft.fft2(in_camera_phase_complex)
        self.dump_spectrum_conv = torch.fft.fft2(phase_complex) * torch.fft.fft2(in_camera_phase_complex)

        conv = torch.fft.ifft2(self.dump_spectrum_conv)  
        #TODUMP
        self.dump_conv_real = conv.real
        self.dump_conv_imag = conv.imag
        #<--------------------------
        return (conv.real ** 2 + conv.imag ** 2)  # n_wl X D X x X y

    # def _psf_at_camera_impl(self, scene_distances, modulate_phase):
    #     '''
    #     Return the 2 cropped version of the psf: cropped psf and full size psf
    #     '''
    #     # As this quadruple will be copied to the other three, rho = 0 is avoided.
    #     psf1d = self.psf1d(scene_distances, modulate_phase).float()
    #     #psf1d = psf1d[..., :self.image_size[0], :self.image_size[1]]
    #     # crop from the center
    #     uncropped_size = psf1d.shape
    #     uncropped_height = uncropped_size[-1]
    #     uncropped_width = uncropped_size[-2]

    #     #croped psf
    #     left = (uncropped_width - self.image_size[0]) // 2
    #     top = (uncropped_height - self.image_size[1]) // 2
    #     right = (uncropped_width + self.image_size[0]) // 2
    #     bottom = (uncropped_height + self.image_size[1]) // 2
    #     cropped_psf1d = psf1d[..., left:right, top:bottom]

    #     #full-size cropped psf
    #     left2 = (uncropped_width - self.full_size[0]) // 2
    #     top2 = (uncropped_height - self.full_size[1]) // 2
    #     right2 = (uncropped_width + self.full_size[0]) // 2
    #     bottom2 = (uncropped_height + self.full_size[1]) // 2
    #     full_psf1d = psf1d[..., left2:right2, top2:bottom2]
    #     return cropped_psf1d, full_psf1d # wl x depth x size x size
    

    def _psf_at_camera_impl(self, scene_distances, modulate_phase):
        # As this quadruple will be copied to the other three, rho = 0 is avoided.
        psf1d = self.psf1d(scene_distances, modulate_phase).float()
        #psf1d = psf1d[..., :self.image_size[0], :self.image_size[1]]
        # crop from the center
        uncropped_size = psf1d.shape
        uncropped_height = uncropped_size[-1]
        uncropped_width = uncropped_size[-2]
        left = (uncropped_width - self.image_size[0]) // 2
        top = (uncropped_height - self.image_size[1]) // 2
        right = (uncropped_width + self.image_size[0]) // 2
        bottom = (uncropped_height + self.image_size[1]) // 2
        psf1d = psf1d[..., left:right, top:bottom]
        return psf1d # wl x depth x size x size

    def _psf_at_camera_impl_full(self, scene_distances, modulate_phase):
        # As this quadruple will be copied to the other three, rho = 0 is avoided.
        psf1d = self.psf1d(scene_distances, modulate_phase).float()
        #psf1d = psf1d[..., :self.image_size[0], :self.image_size[1]]
        # crop from the center
        uncropped_size = psf1d.shape
        uncropped_height = uncropped_size[-1]
        uncropped_width = uncropped_size[-2]
        left = (uncropped_width - self.full_size[0]) // 2
        top = (uncropped_height - self.full_size[1]) // 2
        right = (uncropped_width + self.full_size[0]) // 2
        bottom = (uncropped_height + self.full_size[1]) // 2
        psf1d = psf1d[..., left:right, top:bottom]
        return psf1d # wl x depth x size x size
 
    def to_sensor_phase(self, h_size):
        '''
        The impulse response (h) of free-space propagation tell you how light is spread on the sensor.\n
        The h_size is currently choosen manually.
        '''
        x = self.mask_pitch * torch.arange( - (h_size // 2), h_size // 2 + 1).reshape(-1, 1)
        y = self.mask_pitch * torch.arange( - (h_size // 2), h_size // 2 + 1).reshape(1, -1)
        distance_diff = x ** 2 + y ** 2
        distance_diff = distance_diff.unsqueeze(0).to(self.device)
        z = self.sensor_distance()
        k = (2 * math.pi) / self.wavelengths.reshape(-1, 1, 1).double()
        return k * distance_diff / (2 * z)

    def __padding2size(self, tensor, new_size):
        """
        Padding zeros to the end of a complex-valued tensor.
        """
        x_padding = (new_size - tensor.shape[-1]) 
        # Pad the original tensor to the new size
        real = torch.nn.functional.pad(tensor.real, (0, x_padding, 0, x_padding), mode='constant', value=0)
        imag = torch.nn.functional.pad(tensor.imag, (0, x_padding, 0, x_padding), mode='constant', value=0)
        return torch.complex(real, imag)

    def _capture_impl(self, volume, layered_depth, psf, occlusion, eps=1e-3):
        scale = volume.max()
        volume = volume / scale   # (B, 3, D, H, W)
        #Fpsf = torch.rfft(psf, 2)
        Fpsf = torch.view_as_real(torch.fft.rfft2(psf))
        #if(self.debug):
            # print("fpsf: ")
            # print(Fpsf.shape)
            # print("layered_depth: ")
            # print(layered_depth.shape)
        if occlusion:
            #Fvolume = torch.rfft(volume, 2)
            Fvolume = torch.view_as_real(torch.fft.rfft2(volume))
            #Flayered_depth = torch.rfft(layered_depth, 2)
            Flayered_depth = torch.view_as_real(torch.fft.rfft2(layered_depth))
            # blurred_alpha_rgb = torch.irfft(
            #     complex.multiply(Flayered_depth, Fpsf), 2, signal_sizes=volume.shape[-2:])
            # blurred_volume = torch.irfft(
            #     complex.multiply(Fvolume, Fpsf), 2, signal_sizes=volume.shape[-2:])
            blurred_alpha_rgb = torch.fft.irfft2(torch.view_as_complex(complex.multiply(Flayered_depth, Fpsf)), volume.shape[-2:])
            blurred_volume = torch.fft.irfft2(torch.view_as_complex(complex.multiply(Fvolume, Fpsf)), volume.shape[-2:])
            # Normalize the blurred intensity
            cumsum_alpha = torch.flip(torch.cumsum(torch.flip(layered_depth, dims=(-3,)), dim=-3), dims=(-3,))
            #Fcumsum_alpha = torch.rfft(cumsum_alpha, 2)
            Fcumsum_alpha = torch.view_as_real(torch.fft.rfft2(cumsum_alpha))
            # blurred_cumsum_alpha = torch.irfft(
            #     complex.multiply(Fcumsum_alpha, Fpsf), 2, signal_sizes=volume.shape[-2:])
            blurred_cumsum_alpha = torch.fft.irfft2(torch.view_as_complex(complex.multiply(Fcumsum_alpha, Fpsf)), volume.shape[-2:])
            blurred_volume = blurred_volume / (blurred_cumsum_alpha + eps)
            blurred_alpha_rgb = blurred_alpha_rgb / (blurred_cumsum_alpha + eps)

            over_alpha = utils.helper.over_op(blurred_alpha_rgb)
            captimg = torch.sum(over_alpha * blurred_volume, dim=-3)
        else:
            #Fvolume = torch.rfft(volume, 2)
            Fvolume = torch.view_as_real(torch.fft.rfft2(volume))
            Fcaptimg = complex.multiply(Fvolume, Fpsf).sum(dim=2)
            #captimg = torch.irfft(Fcaptimg, 2, signal_sizes=volume.shape[-2:])
            captimg = torch.fft.irfft2(torch.view_as_complex(Fcaptimg), volume.shape[-2:])

        captimg = scale * captimg
        volume = scale * volume
        return captimg, volume
    
    def dump(self, is_training = False):
        '''
        Dumping features for debugging purpose.
        '''
        dictionary = {}
        device = self.device
        if is_training:
            scene_distances = utils.helper.ips_to_metric(
                torch.linspace(0, 1, steps=self.n_depths, device=device) +
                1 / self.n_depths * (torch.rand(self.n_depths, device=device) - 0.5),
                self.min_depth, self.max_depth)
            scene_distances[-1] += torch.rand(1, device=device)[0] * (100.0 - self.max_depth)
        else:
            scene_distances = utils.helper.ips_to_metric(torch.linspace(0, 1, steps=self.n_depths, device=device),
                                            self.min_depth, self.max_depth)

        #return self.psf1d(scene_distances, torch.tensor(True))

        psf = self.psf1d(scene_distances, torch.tensor(True))
        depth_level = 6
        dictionary['init_phase'] = self.dump_init_phase[0, depth_level]  #real value
        dictionary['init_amplitude'] = self.dump_init_amplitude[0, depth_level]   #real value
        dictionary['mask2D'] = self.dump_mask2D  #real value
        dictionary['amplitude'] = self.dump_amplitude[0, depth_level]   #real value 
        dictionary['phase_complex'] = self.dump_phase_complex.real[0, depth_level] # complex value
        dictionary['h'] = self.dump_h.real[0, 0]  # complex value
        dictionary['h_magnitude'] = torch.log(1 + self.dump_h.real ** 2 + self.dump_h.imag ** 2)[0,0]
        dictionary['phase_complex_fft'] = torch.log(1 + torch.sqrt(self.dump_phase_complex_fft.real[0, depth_level] ** 2 +  self.dump_phase_complex_fft.imag[0, depth_level] ** 2))
        dictionary['h_complex_fft'] = torch.log(1 + torch.sqrt(self.dump_h_complex_fft.imag[0,0] ** 2 + self.dump_h_complex_fft.real[0,0] ** 2)) 
        dictionary['spectrum_conv'] = torch.log(1 + torch.sqrt(self.dump_spectrum_conv.imag[0, depth_level] ** 2 + self.dump_spectrum_conv.real[0, depth_level] ** 2))
        dictionary['conv_real'] = self.dump_conv_real[0, depth_level]  # complex value
        dictionary['conv_imag'] = self.dump_conv_imag[0, depth_level]  # complex value
        dictionary['abs_conv_real'] = torch.abs(self.dump_conv_real[0, depth_level])
        #dictionary['psf'] = torch.log(1e16 + torch.sqrt(psf[0,0])) #real value
        dictionary['psf'] = psf[0,depth_level] #real value
        dictionary['h_prepad'] = self.dump_h_prepad.real[0, 0]
        #dictionary['heightmap'] = self.dump_heightmap2d[0, 0] 
        dictionary['defocus_phase'] = self.dump_defocus_phase[0, depth_level]
        # print('defocus_phase: ')
        # print(self.dump_defocus_phase[0, depth_level])
        dictionary['defocus_ampltide'] = self.dump_defocus_amplitude[0, depth_level]
        dictionary['heightmap_phase'] = self.dump_heightmap2d[0, 0]
        return dictionary

    def dump_depth_psf(self, is_training = False):
        '''
        Dumping every depth level of the psf for debugging purpose.
        '''
        dictionary = {}
        device = self.device
        if is_training:
            scene_distances = utils.helper.ips_to_metric(
                torch.linspace(0, 1, steps=self.n_depths, device=device) +
                1 / self.n_depths * (torch.rand(self.n_depths, device=device) - 0.5),
                self.min_depth, self.max_depth)
            scene_distances[-1] += torch.rand(1, device=device)[0] * (100.0 - self.max_depth)
        else:
            scene_distances = utils.helper.ips_to_metric(torch.linspace(0, 1, steps=self.n_depths, device=device),
                                            self.min_depth, self.max_depth)

        #return self.psf1d(scene_distances, torch.tensor(True))

        psf = self.psf1d(scene_distances, torch.tensor(True))
        for depth_level in range(self.n_depths): 
            name = 'psf_' + str(depth_level)
            dictionary[name] = psf[0,depth_level] #real value
        return dictionary

    def phase_mask_from(self, file : str):  #self, rows, cols
        im = cv2.imread(file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
        return torch.from_numpy(im).to(self.device)

    def circular_mask(self, shape):
        radius = shape // 2
        Y, X = torch.meshgrid(torch.arange(radius), torch.arange(radius))
        mask = (X)**2 + (Y)**2 <= (radius - 1)**2
        return mask.float()

    def copy_quadruple(self, x_rd):
        x_ld = torch.flip(x_rd, dims=(-2,))
        x_d = torch.cat([x_ld, x_rd], dim=-2)
        x_u = torch.flip(x_d, dims=(-1,))
        x = torch.cat([x_u, x_d], dim=-1)
        return x
    
    def test_heightmap_diag(self):
        R1 = self.focal_length / 2
        min_z = 1e10
        Height = self.mask_diameter
        half_mask_size = self.mask_size // 2
        heightmap1d = torch.zeros(half_mask_size)
        for i in range(half_mask_size): 
            x =  math.sqrt(2) * i * Height / self.mask_size
            z = math.sqrt(R1 * R1 - x * x)
            min_z = min(min_z, z)
            heightmap1d[i] = z
        heightmap1d = heightmap1d - min_z
        #return heightmap1d
        return heightmap1d.to(self.device)
    

class MixedCamera(RotationallySymmetricCamera):
    def __init__(self, focal_depth: float, min_depth: float, max_depth: float, n_depths: int,
                 image_size: Union[int, List[int]], mask_size: int, focal_length: float, mask_diameter: float,
                 camera_pixel_pitch: float, wavelengths=[632e-9, 550e-9, 450e-9], mask_upsample_factor=1,
                 diffraction_efficiency=0.7, full_size=100, debug: bool = False, requires_grad: bool = False):
        self.diffraction_efficiency = diffraction_efficiency
        super().__init__(focal_depth, min_depth, max_depth, n_depths, image_size, mask_size, focal_length,
                         mask_diameter, camera_pixel_pitch, wavelengths, mask_upsample_factor, diffraction_efficiency,
                           full_size, debug, requires_grad)
        #initialize DOEs heightmap
        init_heightmap1d = torch.zeros(mask_size // 2 // mask_upsample_factor)  # 1D half size (radius)
        self.heightmap1d_ = torch.nn.Parameter(init_heightmap1d, requires_grad=requires_grad)
        #self.heightmap1d_ = self.test_heightmap_diag()

        #initialize Coded Aperture
        aperture_mask = self.circular_mask(64)
        init_ca2d = torch.zeros(64,64, dtype=torch.float32)
        ca2d = torch.nn.Parameter(init_ca2d, requires_grad=requires_grad)
        self.ca2d = ca2d * aperture_mask

    def build_camera(self):
        H, rho_grid, rho_sampling = self.precompute_H(self.image_size)
        ind = self.find_index(rho_grid, rho_sampling)

        H_full, rho_grid_full, rho_sampling_full = self.precompute_H(self.full_size)
        ind_full = self.find_index(rho_grid_full, rho_sampling_full)

        assert (rho_grid.max(dim=-1)[0] >= rho_sampling.reshape(self.n_wl, -1).max(dim=-1)[0]).all(), \
            'Grid (max): {}, Sampling (max): {}'.format(
                rho_grid.max(dim=-1)[0], rho_sampling.reshape(self.n_wl, -1).max(dim=-1)[0])
        assert (rho_grid.min(dim=-1)[0] <= rho_sampling.reshape(self.n_wl, -1).min(dim=-1)[0]).all(), \
            'Grid (min): {}, Sampling (min): {}'.format(
                rho_grid.min(dim=-1)[0], rho_sampling.reshape(self.n_wl, -1).min(dim=-1)[0])
        self.register_buffer('H', H)
        self.register_buffer('rho_grid', rho_grid)
        self.register_buffer('rho_sampling', rho_sampling)
        self.register_buffer('ind', ind)
        self.register_buffer('H_full', H_full)
        self.register_buffer('rho_grid_full', rho_grid_full)
        # These two parameters are not used for training.
        self.rho_sampling_full = rho_sampling_full
        self.ind_full = ind_full

        self.r_grid, self.r_sampling = self.pre_sampling()
        self.r_ind = self.find_index2(self.r_grid, self.r_sampling)

    def find_index(self, a, v):
        a = a.squeeze(1).cpu().numpy()
        v = v.cpu().numpy()
        index = np.stack([np.searchsorted(a[i, :], v[i], side='left') - 1 for i in range(a.shape[0])], axis=0)
        return torch.from_numpy(index)
    
    def find_index2(self, a, v):
        a = a.cpu().numpy()
        v = v.cpu().numpy()
        index = np.searchsorted(a[:], v, side='right')  
        index = torch.from_numpy(index)
        return torch.where(index > (self.mask_size // 2 + 1), torch.tensor(self.mask_size // 2 + 1), index)

    def heightmap(self):
        heightmap1d = torch.cat([self.heightmap1d().cpu(), torch.zeros((self.mask_size // 2))], dim=0)
        heightmap1d = heightmap1d.reshape(1, 1, -1)
        r_grid = torch.arange(0, self.mask_size, dtype=torch.double)
        y_coord = torch.arange(0, self.mask_size // 2, dtype=torch.double).reshape(-1, 1) + 0.5
        x_coord = torch.arange(0, self.mask_size // 2, dtype=torch.double).reshape(1, -1) + 0.5
        r_coord = torch.sqrt(y_coord ** 2 + x_coord ** 2).unsqueeze(0)
        r_grid = r_grid.reshape(1, -1)
        ind = self.find_index(r_grid, r_coord)
        heightmap11 = utils.helper.cubicspline.interp(r_grid, heightmap1d, r_coord, ind).float()
        heightmap = utils.helper.copy_quadruple(heightmap11).squeeze()
        return heightmap
    
    def heightmap1d(self):
        return F.interpolate(self.heightmap1d_.reshape(1, 1, -1),
                             scale_factor=self.mask_upsample_factor, mode='nearest').reshape(-1)
    def _mask_upscale(self, CA2D):
        '''
        Parameterizing amplitude mask into low-dim n x n resolution.\n
        This function is intend to scale that into the mask resolution.\n
        '''
        new_mask = F.interpolate(CA2D.unsqueeze(0).unsqueeze(0), size=(self.mask_size, self.mask_size), mode='nearest').squeeze()
        return new_mask

    def precompute_H(self, image_size):
        """
        This is assuming that the defocus phase doesn't change much in one pixel.
        Therefore, the mask_size has to be sufficiently large.
        """
        # As this quadruple will be copied to the other three, zero is avoided.
        coord_y = self.camera_pixel_pitch * torch.arange(1, image_size[0] // 2 + 1).reshape(-1, 1)
        coord_x = self.camera_pixel_pitch * torch.arange(1, image_size[1] // 2 + 1).reshape(1, -1)
        coord_y = coord_y.double()
        coord_x = coord_x.double()
        rho_sampling = torch.sqrt(coord_y ** 2 + coord_x ** 2)

        # Avoiding zero as the numerical derivative is not good at zero
        # sqrt(2) is for finding the diagonal of FoV.
        rho_grid = math.sqrt(2) * self.camera_pixel_pitch * (
                torch.arange(-1, max(image_size) // 2 + 1, dtype=torch.double) + 0.5)

        # n_wl x 1 x n_rho_grid
        rho_grid = rho_grid.reshape(1, 1, -1) / (self.wavelengths.reshape(-1, 1, 1).cpu() * self.sensor_distance())
        # n_wl X (image_size[0]//2 + 1) X (image_size[1]//2 + 1)
        rho_sampling = rho_sampling.unsqueeze(0) / (self.wavelengths.reshape(-1, 1, 1).cpu() * self.sensor_distance())

        r = self.mask_pitch * torch.linspace(1, self.mask_size / 2, self.mask_size // 2).double()
        r = r.reshape(1, -1, 1)
        J = torch.where(rho_grid == 0,
                        1 / 2 * r ** 2,
                        1 / (2 * math.pi * rho_grid) * r * scipy.special.jv(1, 2 * math.pi * rho_grid * r))
        h = J[:, 1:, :] - J[:, :-1, :]
        h0 = J[:, 0:1, :]
        return torch.cat([h0, h], dim=1), rho_grid.squeeze(1), rho_sampling

    def precompute2(self, size):
        """
        This is assuming that the defocus phase doesn't change much in one pixel.
        Therefore, the mask_size has to be sufficiently large.
        """
        # As this quadruple will be copied to the other three, zero is avoided.
        coord_y = self.mask_pitch * torch.arange(1, size[0] // 2 + 1).reshape(-1, 1)
        coord_x = self.mask_pitch * torch.arange(1, size[1] // 2 + 1).reshape(1, -1)
        coord_y = coord_y.double()
        coord_x = coord_x.double()
        rho_sampling = torch.sqrt(coord_y ** 2 + coord_x ** 2)
        # Avoiding zero as the numerical derivative is not good at zero
        # sqrt(2) is for finding the diagonal of FoV.
        rho_grid = math.sqrt(2) * self.mask_pitch * (
                torch.arange(-1, max(size) // 2 + 1, dtype=torch.double) + 0.5)
        # n_wl x 1 x n_rho_grid
        rho_grid = rho_grid.reshape(1, 1, -1) / (self.wavelengths.reshape(-1, 1, 1).cpu() * self.sensor_distance())
        # n_wl X (image_size[0]//2 + 1) X (image_size[1]//2 + 1)
        rho_sampling = rho_sampling.unsqueeze(0) / (self.wavelengths.reshape(-1, 1, 1).cpu() * self.sensor_distance())
        return rho_grid.squeeze(1), rho_sampling  

    def pre_sampling(self):
        coord_y = self.mask_pitch * (torch.arange(0, self.mask_size // 2)).reshape(-1, 1)
        coord_x = self.mask_pitch * (torch.arange(0, self.mask_size // 2)).reshape(1, -1)
        coord_y = coord_y.double()
        coord_x = coord_x.double()
        # (mask_size / 2) x (mask_size / 2)
        r_sampling = torch.sqrt(coord_y ** 2 + coord_x ** 2)
        r_sampling =  r_sampling.to(self.device)
        
        r_grid = math.sqrt(2) * self.mask_pitch * (torch.arange(0, self.mask_size // 2 + 1, dtype=torch.double))
        #TOCHANGE
        r_grid = r_grid.to(self.device)
        return r_grid, r_sampling 
    def pointsource_inputfield1d(self, scene_distances):
        device = self.device
        r = self.mask_pitch * torch.linspace(1, self.mask_size / 2, self.mask_size // 2, device=device).double()
        # compute pupil function
        wavelengths = self.wavelengths.reshape(-1, 1, 1).double()
        scene_distances = scene_distances.reshape(1, -1, 1).double()  # 1 x D x 1
        r = r.reshape(1, 1, -1)
        wave_number = 2 * math.pi / wavelengths

        radius = torch.sqrt(scene_distances ** 2 + r ** 2)  # 1 x D x n_r

        # ignore 1/j term (constant phase)
        amplitude = scene_distances / wavelengths / radius ** 2  # n_wl x D x n_r
        amplitude /= amplitude.max()
        # zero phase at center
        phase = wave_number * (radius - scene_distances)  # n_wl x D x n_r
        if not math.isinf(self.focal_depth):
            focal_depth = torch.tensor(self.focal_depth, device=device).reshape(1, 1, 1).double()  # 1 x 1 x 1
            f_radius = torch.sqrt(focal_depth ** 2 + r ** 2)  # 1 x 1 x n_r
            phase -= wave_number * (f_radius - focal_depth)  # subtract focal_depth to roughly remove a piston
        return amplitude, phase

    def psf1d(self, H, scene_distances, modulate_phase=torch.tensor(True)):
        """Perform all computations in double for better precision. Float computation fails."""
        prop_amplitude, prop_phase = self.pointsource_inputfield1d(scene_distances)

        H = H.unsqueeze(1)  # n_wl x 1 x n_r x n_rho
        print(H.shape)
        wavelengths = self.wavelengths.reshape(-1, 1, 1).double()
        if modulate_phase:
            phase_delays = utils.helper.heightmap_to_phase(self.heightmap1d().reshape(1, -1),  # add wavelength dim
                                              wavelengths,
                                              utils.helper.refractive_index(wavelengths))
            phase = phase_delays + prop_phase  # n_wl X D x n_r
        else:
            phase = prop_phase

        # broadcast the matrix-vector multiplication
        phase = utils.helper.copy_quadruple(interp.linterp2(self.r_grid, phase, self.r_sampling, self.r_ind))
        amplitude = utils.helper.copy_quadruple(interp.linterp2(self.r_grid, prop_amplitude, self.r_sampling, self.r_ind))
        amplitude = amplitude * self.ca2d
        # phase = phase.unsqueeze(2)  # n_wl X D X 1 x n_r
        # amplitude = prop_amplitude.unsqueeze(2)  # n_wl X D X 1 x n_r
        # print(self.r_grid.shape)
        # print(phase.shape)
        # print(self.r_sampling.shape)
        # print(self.r_ind.shape)
        # phase = F.relu(utils.interp.interp(self.r_grid, phase, self.r_sampling, self.r_ind).float())
        # phase = phase.reshape(self.n_wl, self.n_depths, self.mask_size // 2, self.mask_size // 2)
        # amplitude = F.relu(utils.interp.interp(self.r_grid, amplitude, self.r_sampling, self.r_ind).float())
        # amplitude = phase.reshape(self.n_wl, self.n_depths, self.mask_size // 2, self.mask_size // 2)
        # amplitude = amplitude * self.ca2d

        real = torch.matmul(amplitude * torch.cos(phase), H).squeeze(-2)
        imag = torch.matmul(amplitude * torch.sin(phase), H).squeeze(-2)
        print(real.shape)
        return (2 * math.pi / wavelengths / self.sensor_distance()) ** 2 * (real ** 2 + imag ** 2)  # n_wl X D X n_rho

    def _psf_at_camera_impl(self, H, rho_grid, rho_sampling, ind, size, scene_distances, modulate_phase):
        # As this quadruple will be copied to the other three, rho = 0 is avoided.
        psf1d = self.psf1d(H, scene_distances, modulate_phase)
        psf_rd = F.relu(utils.interp.interp(rho_grid, psf1d, rho_sampling, ind).float())
        psf_rd = psf_rd.reshape(self.n_wl, self.n_depths, size[0] // 2, size[1] // 2)
        return utils.helper.copy_quadruple(psf_rd)

    def psf_at_camera(self, size=None, modulate_phase=torch.tensor(True), is_training=torch.tensor(False)):
        device = self.device
        if is_training:
            scene_distances = utils.helper.ips_to_metric(
                torch.linspace(0, 1, steps=self.n_depths, device=device) +
                1 / self.n_depths * (torch.rand(self.n_depths, device=device) - 0.5),
                self.min_depth, self.max_depth)
            scene_distances[-1] += torch.rand(1, device=device)[0] * (100.0 - self.max_depth)
        else:
            scene_distances = utils.helper.ips_to_metric(torch.linspace(0, 1, steps=self.n_depths, device=device),
                                            self.min_depth, self.max_depth)

        diffracted_psf = self._psf_at_camera_impl(
            self.H, self.rho_grid, self.rho_sampling, self.ind, self.image_size, scene_distances, modulate_phase)
        undiffracted_psf = self._psf_at_camera_impl(
            self.H, self.rho_grid, self.rho_sampling, self.ind, self.image_size, scene_distances, torch.tensor(False))

        # Keep the normalization factor for penalty computation
        self.diff_normalization_scaler = diffracted_psf.sum(dim=(-1, -2), keepdim=True)
        self.undiff_normalization_scaler = undiffracted_psf.sum(dim=(-1, -2), keepdim=True)

        diffracted_psf = diffracted_psf / self.diff_normalization_scaler
        undiffracted_psf = undiffracted_psf / self.undiff_normalization_scaler

        psf = self.diffraction_efficiency * diffracted_psf + (1 - self.diffraction_efficiency) * undiffracted_psf

        # In training, randomly pixel-shifts the PSF around green channel.
        if is_training:
            max_shift = 2
            r_shift = tuple(np.random.randint(low=-max_shift, high=max_shift, size=2))
            b_shift = tuple(np.random.randint(low=-max_shift, high=max_shift, size=2))
            psf_r = torch.roll(psf[0], shifts=r_shift, dims=(-1, -2))
            psf_g = psf[1]
            psf_b = torch.roll(psf[2], shifts=b_shift, dims=(-1, -2))
            psf = torch.stack([psf_r, psf_g, psf_b], dim=0)

        if torch.tensor(size is not None):
            pad_h = (size[0] - self.image_size[0]) // 2
            pad_w = (size[1] - self.image_size[1]) // 2
            psf = F.pad(psf, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
        return fftshift(psf, dims=(-1, -2))

    def psf_out_of_fov_energy(self, psf_size: int):
        """This can be run only after psf_at_camera is evaluated once."""
        device = self.H.device
        scene_distances = utils.helper.ips_to_metric(torch.linspace(0, 1, steps=self.n_depths, device=device),
                                        self.min_depth, self.max_depth)
        psf1d_diffracted = self.psf1d_full(scene_distances, torch.tensor(True))
        # Normalize PSF based on the cropped PSF
        psf1d_diffracted = psf1d_diffracted / self.diff_normalization_scaler.squeeze(-1)
        edge = psf_size / 2 * self.camera_pixel_pitch / (
                self.wavelengths.reshape(-1, 1, 1) * self.sensor_distance())
        psf1d_out_of_fov = psf1d_diffracted * (self.rho_grid_full.unsqueeze(1) > edge).float()
        return psf1d_out_of_fov.sum(), psf1d_out_of_fov.max()

    def psf1d_full(self, scene_distances, modulate_phase=torch.tensor(True)):
        return self.psf1d(self.H_full, scene_distances, modulate_phase=modulate_phase)

    def forward_train(self, img, depthmap, occlusion):
        return self.forward(img, depthmap, occlusion, is_training=torch.tensor(True))

    def set_diffraction_efficiency(self, de: float):
        self.diffraction_efficiency = de
        self.build_camera()
    def get_psf(self):
        device = self.device
        is_training = False
        if is_training:
            scene_distances = utils.helper.ips_to_metric(
                torch.linspace(0, 1, steps=self.n_depths, device=device) +
                1 / self.n_depths * (torch.rand(self.n_depths, device=device) - 0.5),
                self.min_depth, self.max_depth)
            scene_distances[-1] += torch.rand(1, device=device)[0] * (100.0 - self.max_depth)
        else:
            scene_distances = utils.helper.ips_to_metric(torch.linspace(0, 1, steps=self.n_depths, device=device),
                                            self.min_depth, self.max_depth)

        return self._psf_at_camera_impl(
            self.H, self.rho_grid, self.rho_sampling, self.ind, self.image_size, scene_distances,torch.tensor(True)) 

    def dump(self, is_training = False):
        '''
        Dumping features for debugging purpose.
        '''
        dictionary = {}
        device = self.device
        if is_training:
            scene_distances = utils.helper.ips_to_metric(
                torch.linspace(0, 1, steps=self.n_depths, device=device) +
                1 / self.n_depths * (torch.rand(self.n_depths, device=device) - 0.5),
                self.min_depth, self.max_depth)
            scene_distances[-1] += torch.rand(1, device=device)[0] * (100.0 - self.max_depth)
        else:
            scene_distances = utils.helper.ips_to_metric(torch.linspace(0, 1, steps=self.n_depths, device=device),
                                            self.min_depth, self.max_depth)

        #return self.psf1d(scene_distances, torch.tensor(True))

        #psf = self.psf1d(scene_distances, torch.tensor(True))
        psf = self._psf_at_camera_impl(
            self.H, self.rho_grid, self.rho_sampling, self.ind, self.image_size, scene_distances, torch.tensor(True))
        depth_level = 6
        #dictionary['init_phase'] = self.dump_init_phase[0, depth_level]  #real value
        #dictionary['init_amplitude'] = self.dump_init_amplitude[0, depth_level]   #real value
        #dictionary['mask2D'] = self.dump_mask2D  #real value
        #dictionary['amplitude'] = self.dump_amplitude[0, depth_level]   #real value 
        #dictionary['phase_complex'] = self.dump_phase_complex.real[0, depth_level] # complex value
        #dictionary['h'] = self.dump_h.real[0, 0]  # complex value
        #dictionary['h_magnitude'] = torch.log(1 + self.dump_h.real ** 2 + self.dump_h.imag ** 2)[0,0]
        #dictionary['phase_complex_fft'] = torch.log(1 + torch.sqrt(self.dump_phase_complex_fft.real[0, depth_level] ** 2 +  self.dump_phase_complex_fft.imag[0, depth_level] ** 2))
        #dictionary['h_complex_fft'] = torch.log(1 + torch.sqrt(self.dump_h_complex_fft.imag[0,0] ** 2 + self.dump_h_complex_fft.real[0,0] ** 2)) 
        #dictionary['spectrum_conv'] = torch.log(1 + torch.sqrt(self.dump_spectrum_conv.imag[0, depth_level] ** 2 + self.dump_spectrum_conv.real[0, depth_level] ** 2))
        #dictionary['conv_real'] = self.dump_conv_real[0, depth_level]  # complex value
        #dictionary['conv_imag'] = self.dump_conv_imag[0, depth_level]  # complex value
        #dictionary['abs_conv_real'] = torch.abs(self.dump_conv_real[0, depth_level])
        #dictionary['psf'] = torch.log(1e16 + torch.sqrt(psf[0,0])) #real value
        dictionary['psf'] = psf[0,depth_level] #real value
        #dictionary['h_prepad'] = self.dump_h_prepad.real[0, 0]
        #dictionary['heightmap'] = self.dump_heightmap2d[0, 0] 
        #dictionary['defocus_phase'] = self.dump_defocus_phase[0, depth_level]
        # print('defocus_phase: ')
        # print(self.dump_defocus_phase[0, depth_level])
        #dictionary['defocus_ampltide'] = self.dump_defocus_amplitude[0, depth_level]
        #dictionary['heightmap_phase'] = self.dump_heightmap2d[0, 0]
        return dictionary

    def dump_depth_psf(self, is_training = False):
        '''
        Dumping every depth level of the psf for debugging purpose.
        '''
        dictionary = {}
        device = self.device
        if is_training:
            scene_distances = utils.helper.ips_to_metric(
                torch.linspace(0, 1, steps=self.n_depths, device=device) +
                1 / self.n_depths * (torch.rand(self.n_depths, device=device) - 0.5),
                self.min_depth, self.max_depth)
            scene_distances[-1] += torch.rand(1, device=device)[0] * (100.0 - self.max_depth)
        else:
            scene_distances = utils.helper.ips_to_metric(torch.linspace(0, 1, steps=self.n_depths, device=device),
                                            self.min_depth, self.max_depth)

        #return self.psf1d(scene_distances, torch.tensor(True))

        #psf = self.psf1d(scene_distances, torch.tensor(True))
        psf = self._psf_at_camera_impl(
            self.H, self.rho_grid, self.rho_sampling, self.ind, self.image_size, scene_distances, torch.tensor(True))
        for depth_level in range(self.n_depths): 
            name = 'psf_' + str(depth_level)
            dictionary[name] = psf[0,depth_level] #real value
            #dictionary[name] = - torch.log(psf[0,depth_level]) #real value
        print("this psf:")
        print(torch.log( psf[0, 15]+ 1 ))
        return dictionary  
    
    def test_heightmap_diag(self):
        R1 = self.focal_length / 2
        min_z = 1e10
        Height = self.mask_diameter
        half_mask_size = self.mask_size // 2
        heightmap1d = torch.zeros(half_mask_size)
        for i in range(half_mask_size): 
            x =  math.sqrt(2) * i * Height / self.mask_size
            z = math.sqrt(R1 * R1 - x * x)
            min_z = min(min_z, z)
            heightmap1d[i] = z
        heightmap1d = heightmap1d - min_z
        #return heightmap1d
        return heightmap1d.to(self.device)
    
    
    def circular_mask(self, shape):
        radius = shape // 2
        Y, X = torch.meshgrid(torch.arange(radius), torch.arange(radius))
        mask = (X)**2 + (Y)**2 <= (radius - 1)**2
        x_rd = mask.float()
        x_ld = torch.flip(x_rd, dims=(-2,))
        x_d = torch.cat([x_ld, x_rd], dim=-2)
        x_u = torch.flip(x_d, dims=(-1,))
        x = torch.cat([x_u, x_d], dim=-1)
        return x