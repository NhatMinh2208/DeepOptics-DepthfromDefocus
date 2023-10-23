import abc
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.special
import utils.helper
from utils import complex, interp
from utils.fft import fftshift
class Camera(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, focal_depth, min_depth, max_depth, n_depths, image_size, mask_size,
                 focal_length, mask_diameter, camera_pixel_pitch, wavelengths, mask_upsample_factor=1, diffraction_efficiency=0.7,
                 full_size=100, requires_grad = False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert min_depth > 1e-6, f'Minimum depth is too small. min_depth: {min_depth}'
        #Creates a one-dimensional tensor of size steps whose values are evenly spaced from start to end, inclusive. 
        scene_distances = utils.helper.ips_to_metric(torch.linspace(0, 1, steps=n_depths), min_depth, max_depth)

        self.n_depths = len(scene_distances)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.focal_depth = focal_depth
        self.focal_length = focal_length
        self.mask_diameter = mask_diameter

        #the distance between holes in the shadow mask. ex: distance between pĩxel
        self.mask_pitch = self.mask_diameter / mask_size
        self.camera_pixel_pitch = camera_pixel_pitch
        #number of pixel per diameter
        self.mask_size = mask_size

        self.image_size = self._normalize_image_size(image_size)
        self._register_wavlength(wavelengths)
        self.diffraction_efficiency = diffraction_efficiency

        # init_heightmap1d = torch.zeros(mask_size // 2 // mask_upsample_factor)  # 1D half size (radius)
        # self.heightmap1d_ = torch.nn.Parameter(init_heightmap1d, requires_grad=requires_grad)
        self.mask_upsample_factor = mask_upsample_factor

        # #TOCHANGE
        # init_ampmask2d = torch.zeros(8,8)
        # self.ampmask2d = torch.nn.Parameter(init_ampmask2d, requires_grad=requires_grad)

        self.full_size = self._normalize_image_size(full_size)
        # self.full_size = [32, 32]
        self.register_buffer('scene_distances', scene_distances)

        self.build_camera()
    # def build_camera(self):
    #     prop_amplitude, prop_phase = self.pointsource_inputfield1d()

    #     H, rho_grid, rho_sampling = self.precompute_H(self.image_size)
    #     ind = self.find_index(rho_grid, rho_sampling)

    #     H_full, rho_grid_full, rho_sampling_full = self.precompute_H(self.full_size)
    #     ind_full = self.find_index(rho_grid_full, rho_sampling_full)

    #     assert (rho_grid.max(dim=-1)[0] >= rho_sampling.reshape(self.n_wl, -1).max(dim=-1)[0]).all(), \
    #         'Grid (max): {}, Sampling (max): {}'.format(
    #             rho_grid.max(dim=-1)[0], rho_sampling.reshape(self.n_wl, -1).max(dim=-1)[0])
    #     assert (rho_grid.min(dim=-1)[0] <= rho_sampling.reshape(self.n_wl, -1).min(dim=-1)[0]).all(), \
    #         'Grid (min): {}, Sampling (min): {}'.format(
    #             rho_grid.min(dim=-1)[0], rho_sampling.reshape(self.n_wl, -1).min(dim=-1)[0])

    #     self.register_buffer('prop_amplitude', prop_amplitude)
    #     self.register_buffer('prop_phase', prop_phase)

    #     self.register_buffer('H', H)
    #     self.register_buffer('rho_grid', rho_grid)
    #     self.register_buffer('rho_sampling', rho_sampling)
    #     self.register_buffer('ind', ind)

    #     self.register_buffer('H_full', H_full)
    #     self.register_buffer('rho_grid_full', rho_grid_full)
    #     # These two parameters are not used for training.
    #     self.rho_sampling_full = rho_sampling_full
    #     self.ind_full = ind_full

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
        #TOCHANGE
        rho_sampling = rho_sampling.to(self.device)


        # Avoiding zero as the numerical derivative is not good at zero
        # sqrt(2) is for finding the diagonal of FoV.
        rho_grid = math.sqrt(2) * self.camera_pixel_pitch * (
                torch.arange(-1, max(image_size) // 2 + 1, dtype=torch.double) + 0.5)

        #TOCHANGE
        rho_grid = rho_grid.to(self.device)
        #

        # n_wl x 1 x n_rho_grid
        rho_grid = rho_grid.reshape(1, 1, -1) / (self.wavelengths.reshape(-1, 1, 1) * self.sensor_distance())
        # n_wl X (image_size[0]//2 + 1) X (image_size[1]//2 + 1)
        rho_sampling = rho_sampling.unsqueeze(0) / (self.wavelengths.reshape(-1, 1, 1) * self.sensor_distance())

        # mask radius
        r = self.mask_pitch * torch.linspace(1, self.mask_size / 2, self.mask_size // 2).double()
        r = r.reshape(1, -1, 1)
        
        #TOCHANGE
        r = r.to(self.device)

        #TOCHANGE
        #Bessel function of the first kind of real order and complex argument
        # J = torch.where(rho_grid == 0,
        #                 1 / 2 * r ** 2,
        #                 1 / (2 * math.pi * rho_grid) * r * scipy.special.jv(1, 2 * math.pi * rho_grid * r)) # n_wl X n_r X n_rho_grid
        J = torch.where(rho_grid == 0,
                        1 / 2 * r ** 2,
                        1 / (2 * math.pi * rho_grid) * r * torch.special.bessel_j1(2 * math.pi * rho_grid * r)) # n_wl X n_r X n_rho_grid
        h = J[:, 1:, :] - J[:, :-1, :]
        h0 = J[:, 0:1, :]
        #return # n_wl X n_r X n_rho_grid, n_wl X n_rho_grid, n_wl X (image_size[0]//2 + 1) X (image_size[1]//2 + 1)
        return torch.cat([h0, h], dim=1), rho_grid.squeeze(1), rho_sampling

    def pointsource_inputfield1d(self):
        # It generates evenly spaced values between 1 and self.mask_size / 2, with self.mask_size // 2 elements in total. 
        # This will create a sequence of values like [1, 2, 3, ..., self.mask_size / 2]
        # pixel pitch(size) * pixel index
        r = self.mask_pitch * torch.linspace(1, self.mask_size / 2, self.mask_size // 2).double()
        # compute pupil function

        #.reshape(-1, 1, 1)
        # The -1 in the first dimension is a placeholder that is automatically inferred based on the total number of elements. 
        # The 1 in the second and third dimensions specifies that those dimensions should have size 1
        wavelength = self.wavelengths.reshape(-1, 1, 1).double() # W x 1 x 1 : Wavelength
        z = self.scene_distances.reshape(1, -1, 1).double()  # 1 x D x 1 : Depth level
        r = r.reshape(1, 1, -1) # 1 x 1 x R : Aperture radius
        wave_number = 2 * math.pi / wavelength

        sqrt_z_r = torch.sqrt(z ** 2 + r ** 2)  # 1 x D x R

        # ignore 1/j term (constant phase)
        amplitude = z / wavelength / sqrt_z_r ** 2  # W x D x R
        amplitude /= amplitude.max()
        # zero phase at center
        phase = wave_number * (sqrt_z_r - z)  # W x D x R
        if not math.isinf(self.focal_depth): #focal_depth: d
            d = torch.tensor(self.focal_depth).reshape(1, 1, 1).double()  # 1 x 1 x 1
            sqrt_r_d = torch.sqrt(d ** 2 + r ** 2)  # 1 x 1 x R
            phase -= wave_number * (sqrt_r_d - d)  # subtract focal_depth to roughly remove a piston
        return amplitude, phase

   
    def normalize_psf(self, psfimg):
        # Scale the psf
        # As the incoming light doesn't change, we compute the PSF energy without the phase modulation
        # and use it to normalize PSF with phase modulation.
        return psfimg / psfimg.sum(dim=(-2, -1), keepdims=True)
        # n_wl X D x H x W  / n_wl x D x 1 x 1 -> n_wl X D x H x W
        # each pixel divide by the sum of whole image pixel
    
    def psf1d(self, H, modulate_phase=torch.tensor(True)):
        """Perform all computations in double for better precision. Float computation fails."""
        H = H.unsqueeze(1)  # n_wl x 1 x n_r x n_rho
        wavelengths = self.wavelengths.reshape(-1, 1, 1).double()
        if modulate_phase:
            phase_delays = utils.helper.heightmap_to_phase(self.heightmap1d().reshape(1, -1),  # add wavelength dim
                                              wavelengths,
                                              utils.helper.refractive_index(wavelengths))

            phase = phase_delays + self.prop_phase  # n_wl X D x n_r
        else:
            phase = self.prop_phase

        # broadcast the matrix-vector multiplication
        phase = phase.unsqueeze(2)  # n_wl X D X 1 x n_r
        amplitude = self.prop_amplitude.unsqueeze(2)  # n_wl X D X 1 x n_r
        real = torch.matmul(amplitude * torch.cos(phase), H).squeeze(-2)  # n_wl X D X n_rho
        imag = torch.matmul(amplitude * torch.sin(phase), H).squeeze(-2)  # n_wl X D X n_rho

        return (2 * math.pi / wavelengths / self.sensor_distance()) ** 2 * (real ** 2 + imag ** 2)  # n_wl X D X n_rho

    def _psf_at_camera_impl(self, H, rho_grid, rho_sampling, ind, size, modulate_phase):
        # As this quadruple will be copied to the other three, rho = 0 is avoided.
        psf1d = self.psf1d(H, modulate_phase)
        # psf1d: n_wl X D X n_rho
        # rho_grid: n_wl X n_rho
        # rho_sampling: n_wl X (n_rho - 2) X (n_rho - 2)
        # ind: n_wl X (n_rho - 2) X (n_rho - 2)
        psf_rd = F.relu(interp.interp(rho_grid, psf1d, rho_sampling, ind).float())
        psf_rd = psf_rd.reshape(self.n_wl, self.n_depths, size[0] // 2, size[1] // 2)
        return utils.helper.copy_quadruple(psf_rd)

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
        #Fpsf = torch.rfft(psf, 2)
        Fpsf = torch.view_as_real(torch.fft.rfft2(psf))
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
                 full_size=100, requires_grad = False):
        super().__init__(focal_depth, min_depth, max_depth, n_depths, image_size, mask_size,
                 focal_length, mask_diameter, camera_pixel_pitch, wavelengths, mask_upsample_factor, diffraction_efficiency,
                 full_size, requires_grad)
        init_heightmap1d = torch.zeros(mask_size // 2 // mask_upsample_factor)  # 1D half size (radius)
        self.heightmap1d_ = torch.nn.Parameter(init_heightmap1d, requires_grad=requires_grad)

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

    def heightmap1d(self):
        #F.interpolate(..., mode='nearest'): Applies nearest-neighbor interpolation to the 3D tensor obtained in the previous step. 
        # The F.interpolate function is often used for upsampling or downsampling tensors. In this case,
        #  it seems like it's being used to change the size of the tensor.
        #.reshape(-1): flatten tensor to 1D
        return F.interpolate(self.heightmap1d_.reshape(1, 1, -1),
                             scale_factor=self.mask_upsample_factor, mode='nearest').reshape(-1)
        #i think this made nothing

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
        return self.psf1d(self.H_full, scene_distances, modulate_phase=modulate_phase)
    
    def _psf_at_camera_impl(self, H, rho_grid, rho_sampling, ind, size, scene_distances, modulate_phase):
        # As this quadruple will be copied to the other three, rho = 0 is avoided.
        psf1d = self.psf1d(H, scene_distances, modulate_phase)
        psf_rd = F.relu(interp.interp(rho_grid, psf1d, rho_sampling, ind).float())
        psf_rd = psf_rd.reshape(self.n_wl, self.n_depths, size[0] // 2, size[1] // 2)
        return utils.helper.copy_quadruple(psf_rd) # wl x depth x size x size

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
        print("undiffracted_psf.shape: ") #torch.Size([3, 16, 384, 384])
        print(undiffracted_psf.shape)
        print(undiffracted_psf.dtype)
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
    
class AsymmetricMaskRotationallySymmetricCamera(RotationallySymmetricCamera):
    def __init__(self, focal_depth, min_depth, max_depth, n_depths, image_size, mask_size,
                 focal_length, mask_diameter, camera_pixel_pitch, wavelengths, mask_upsample_factor=1, diffraction_efficiency=0.7,
                 full_size=100, requires_grad = False):
        super().__init__(focal_depth, min_depth, max_depth, n_depths, image_size, mask_size,
                 focal_length, mask_diameter, camera_pixel_pitch, wavelengths, mask_upsample_factor, diffraction_efficiency,
                 full_size, requires_grad)
         #TOCHANGE
        init_ampmask2d = torch.ones(8,8)
        self.ampmask2d = torch.nn.Parameter(init_ampmask2d, requires_grad=requires_grad)

    def build_camera(self):
        H, _, _ = self.precompute_H(self.image_size)
        # ind = self.find_index(rho_grid, rho_sampling)

        # H_full, rho_grid_full, rho_sampling_full = self.precompute_H(self.full_size)
        # ind_full = self.find_index(rho_grid_full, rho_sampling_full)

        # #TOCHANGE
        # self.r_grid, self.r_sampling = self.radius_sampling()
        # self.r_ind = self.find_index(self.r_grid, self.r_sampling)

        # assert (rho_grid.max(dim=-1)[0] >= rho_sampling.reshape(self.n_wl, -1).max(dim=-1)[0]).all(), \
        #     'Grid (max): {}, Sampling (max): {}'.format(
        #         rho_grid.max(dim=-1)[0], rho_sampling.reshape(self.n_wl, -1).max(dim=-1)[0])
        # assert (rho_grid.min(dim=-1)[0] <= rho_sampling.reshape(self.n_wl, -1).min(dim=-1)[0]).all(), \
        #     'Grid (min): {}, Sampling (min): {}'.format(
        #         rho_grid.min(dim=-1)[0], rho_sampling.reshape(self.n_wl, -1).min(dim=-1)[0])
        self.register_buffer('H', H)
        # self.register_buffer('rho_grid', rho_grid)
        # self.register_buffer('rho_sampling', rho_sampling)
        # self.register_buffer('ind', ind)
        # self.register_buffer('H_full', H_full)
        # self.register_buffer('rho_grid_full', rho_grid_full)
        # # These two parameters are not used for training.
        # self.rho_sampling_full = rho_sampling_full
        # self.ind_full = ind_full

        self.rho_grid, self.rho_sampling = self._rho_sampling()
        self.rho_index = self.find_index(self.rho_grid, self.rho_sampling)
        print(self.rho_grid.shape)
        print(self.rho_index)

        self.ind = self.rho_index
    def find_index(self, a, v):
        #The find_index function you've provided takes two tensors, a and v, and returns a tensor of indices where 
        #each index corresponds to the leftmost position in a where the corresponding element in v can be inserted
        #while maintaining the sorted order.
        a = a.cpu().numpy()
        v = v.cpu().numpy()
        index = np.searchsorted(a[:], v, side='left') - 1 
        return torch.from_numpy(index)
    
    def heightmap_(self):
        heightmap1d = torch.cat([self.heightmap1d().to(self.device), torch.zeros((self.full_size[0] // 2)).to(self.device)], dim=0)
        heightmap1d = heightmap1d.reshape(1, 1, -1)
        r_grid = torch.arange(0, self.full_size[0], dtype=torch.double)
        y_coord = torch.arange(0, self.full_size[0] // 2, dtype=torch.double).reshape(-1, 1) + 0.5
        x_coord = torch.arange(0, self.full_size[0] // 2, dtype=torch.double).reshape(1, -1) + 0.5
        r_coord = torch.sqrt(y_coord ** 2 + x_coord ** 2).unsqueeze(0)
        r_grid = r_grid.reshape(1, -1)
        ind = self.find_index(r_grid, r_coord)
        heightmap11 = interp.interp(r_grid, heightmap1d, r_coord, ind).float()
        heightmap = utils.helper.copy_quadruple(heightmap11).squeeze()
        return heightmap
    
    def mask_upscale(self):
        #upscale to mask size
        tanh_mask2D = torch.tanh(self.ampmask2d)
        new_mask =  F.interpolate(tanh_mask2D.unsqueeze(0).unsqueeze(0), size=(self.mask_size, self.mask_size), mode='nearest').squeeze()
        return new_mask
    
    def _mask_upscale(self):
        #upscale to mask size
        tanh_mask2D = torch.tanh(self.ampmask2d)
        new_mask =  F.interpolate(tanh_mask2D.unsqueeze(0).unsqueeze(0), size=(self.image_size[0], self.image_size[1]), mode='nearest').squeeze()
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

    def rescale2D(self, param1D, grid, sampling, ind):
        # param1d: a X b X n_r
        # grid: a X n_r
        # sampling: a X (n_r - 2) X (n_r - 2)
        # ind: a X (n_r - 2) X (n_r - 2)
        # print("\nparam1D: ")
        # print(param1D.shape)
        # print("\ngrid: ")
        # print(grid.shape)
        # print("\nsampling: ")
        # print(sampling.shape)
        # print("\nind: ")
        # print(ind.shape)
        param2D = F.relu(interp.interp(grid, param1D, sampling, ind).double())
        #param2D = param2D.reshape(param2D.size(0), param2D.size(1), size // 2, size // 2)
        return utils.helper.copy_quadruple(param2D)
    
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

    def precompute_H(self, image_size):
        """
        This is assuming that the defocus phase doesn't change much in one pixel.
        Therefore, the mask_size has to be sufficiently large.
        """
        # As this quadruple will be copied to the other three, zero is avoided.
        coord_y = self.camera_pixel_pitch * torch.arange(1, image_size[0] // 2 + 1).reshape(-1, 1)
        coord_x = self.camera_pixel_pitch * torch.arange(1, image_size[1] // 2 + 1 ).reshape(1, -1)
        coord_y = coord_y.double()
        coord_x = coord_x.double()
        rho_sampling = torch.sqrt(coord_y ** 2 + coord_x ** 2)
        #TOCHANGE
        rho_sampling = rho_sampling.to(self.device)


        # Avoiding zero as the numerical derivative is not good at zero
        # sqrt(2) is for finding the diagonal of FoV.
        rho_grid = math.sqrt(2) * self.camera_pixel_pitch * (
                torch.arange(0, max(image_size) // 2 + 1 , dtype=torch.double) + 0.5)

        #TOCHANGE
        rho_grid = rho_grid.to(self.device)
        #

        # n_wl x 1 x n_rho_grid
        rho_grid = rho_grid.reshape(1, 1, -1) / (self.wavelengths.reshape(-1, 1, 1) * self.sensor_distance())
        # n_wl X (image_size[0]//2 + 1) X (image_size[1]//2 + 1)
        rho_sampling = rho_sampling.unsqueeze(0) / (self.wavelengths.reshape(-1, 1, 1) * self.sensor_distance())

        # mask radius
        r = self.mask_pitch * torch.linspace(1, self.mask_size / 2, self.mask_size // 2).double()
        r = r.reshape(1, -1, 1)
        
        #TOCHANGE
        r = r.to(self.device)

        #TOCHANGE
        #Bessel function of the first kind of real order and complex argument
        # J = torch.where(rho_grid == 0,
        #                 1 / 2 * r ** 2,
        #                 1 / (2 * math.pi * rho_grid) * r * scipy.special.jv(1, 2 * math.pi * rho_grid * r)) # n_wl X n_r X n_rho_grid
        J = torch.where(rho_grid == 0,
                        1 / 2 * r ** 2,
                        1 / (2 * math.pi * rho_grid) * r * torch.special.bessel_j1(2 * math.pi * rho_grid * r)) # n_wl X n_r X n_rho_grid
        h = J[:, 1:, :] - J[:, :-1, :]
        h0 = J[:, 0:1, :]
        #return # n_wl X n_r X n_rho_grid, n_wl X n_rho_grid, n_wl X (image_size[0]//2 + 1) X (image_size[1]//2 + 1)
        return torch.cat([h0, h], dim=1), rho_grid.squeeze(1), rho_sampling
    
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

    def psf1d(self, scene_distances, modulate_phase=torch.tensor(True)):
        """Perform all computations in double for better precision. Float computation fails."""
        prop_amplitude, prop_phase = self.pointsource_inputfield1d(scene_distances) # n_wl X D x r
        amplitude = prop_amplitude
        #H = self.rescale2D(H, self.r_grid, self.r_sampling, self.r_ind, self.mask_diameter)
        # H = H.unsqueeze(1)  # n_wl x 1 x H x W x n_rho
        in_camera_phase = self.to_sensor_phase().unsqueeze(1) # n_wl x 1 x X x y
        wavelengths = self.wavelengths.reshape(-1, 1, 1).double()
        if modulate_phase:
            # heightmap size: (mask_size // 2 // mask_upsample_factor). In this case choose upscale factor = 1.
            phase_delays = utils.helper.heightmap_to_phase(self.heightmap1d().reshape(1, -1),  # add wavelength dim
                                              wavelengths,
                                              utils.helper.refractive_index(wavelengths))  #n_wl X r
            #prop_phase: n_wl X D x r  (r : mask_size // 2)
            phase = phase_delays + prop_phase  # n_wl X D x r
                                               # desired: n_wl x D x x x y        
        else:
            phase = prop_phase

        # phase = F.interpolate(phase, size=self.image_size[0], mode='linear') # n_wl x D x n_rho
        # amplitude = F.interpolate(amplitude, size=self.image_size[0], mode='linear') # n_wl x D x n_rho
        phase = self.mask2sensor_scale(phase) # n_wl x D x n_rho
        amplitude = self.mask2sensor_scale(amplitude) # n_wl x D x n_rho
        phase = interp.linterp(self.rho_grid, phase, self.rho_sampling, self.rho_index) # n_wl x D x X x Y  
        amplitude = interp.linterp(self.rho_grid, amplitude, self.rho_sampling, self.rho_index)  # n_wl x D x X x Y
        phase = utils.helper.copy_quadruple(phase)
        amplitude = utils.helper.copy_quadruple(amplitude)
        mask2D = self._mask_upscale() # x X y  
        amplitude = amplitude * mask2D # n_wl x D x X x Y
        # phase = self.mask2pixel_size(phase) #n_wl x D x X x Y  (X, Y is pixel coordination)
        # amplitude = self.mask2pixel_size(amplitude) #n_wl x D x X x Y     
 
        
        # phase = self.rescale2D(phase, self.r_grid, self.r_sampling, self.r_ind, self.mask_diameter) # desired: n_wl x D x x_prime x y_prime   
        # phase = self.rescale2D(phase, self.rho_grid, self.rho_sampling, self.ind) # desired: n_wl x D x x x y   
        # amplitude = self.rescale2D(amplitude, self.rho_grid, self.rho_sampling, self.ind) # desired: n_wl x D x x_prime x y_prime   
        # broadcast the matrix-vector multiplication
        # phase = phase.unsqueeze(2)  # n_wl X D X 1 x x_prime x y_prime   
        # amplitude = amplitude.unsqueeze(2)  # n_wl X D X 1 x x_prime x y_prime   
        # in_camera_phase = self.__reshaping(in_camera_phase)
        # phase = phase.reshape(phase.size(0), phase.size(1), phase.size(2), -1)
        # amplitude = amplitude.reshape(amplitude.size(0), amplitude.size(1), amplitude.size(2), -1)
        print("amplitude: ")
        print(amplitude)

        phase_complex = torch.complex(amplitude * torch.cos(phase), amplitude * torch.sin(phase))
        in_camera_phase_complex = torch.complex(torch.cos(in_camera_phase), torch.sin(in_camera_phase))
        some_constant1 = ( (2 * math.pi) / self.wavelengths.reshape(-1, 1) ) / self.scene_distances.reshape(1, -1) # n_wl X D
        some_constant1 = torch.complex(torch.cos(some_constant1), torch.sin(some_constant1))
        some_constant2 = torch.complex(torch.zeros((self.wavelengths.shape[0], self.scene_distances.shape[0])).to(self.device) , self.wavelengths.reshape(-1, 1) * self.scene_distances.reshape(1, -1) ) # n_wl X D
        in_camera_phase_complex = (some_constant1.unsqueeze(2).unsqueeze(3) / some_constant2.unsqueeze(2).unsqueeze(3)) * torch.complex(torch.cos(in_camera_phase), torch.sin(in_camera_phase))
        # print("\nphase_complex: ")
        # print(phase_complex.shape)
        # print("\nin_camera_phase_complex: ")
        # print(in_camera_phase_complex.shape)
        conv = torch.fft.ifft2(torch.fft.fft2(phase_complex) * torch.fft.fft2(in_camera_phase_complex), phase.shape[-2:])  # n_wl X D X x X y (x,y: complex number)
        print(conv.shape)
        # print("phase_complex: ")
        # print(phase_complex)
        # print("in_camera_phase_complex: ")
        # print(in_camera_phase_complex)
        # real = torch.matmul(amplitude * torch.cos(phase), torch.cos(in_camera_phase)).squeeze(-2)
        # - torch.matmul(amplitude * torch.sin(phase), torch.sin(in_camera_phase)).squeeze(-2)  # n_wl X D X x X y
        # imag = torch.matmul(amplitude * torch.sin(phase), torch.cos(in_camera_phase)).squeeze(-2)
        # + torch.matmul(amplitude * torch.cos(phase), torch.sin(in_camera_phase)).squeeze(-2)  # n_wl X D X x X y
        # print(real.shape)
        return (conv.real ** 2 + conv.imag ** 2)  # n_wl X D X x X y
    
    def _psf_at_camera_impl(self, H, rho_grid, rho_sampling, ind, size, scene_distances, modulate_phase):
        # As this quadruple will be copied to the other three, rho = 0 is avoided.
        psf1d = self.psf1d(scene_distances, modulate_phase).float()
        # print(psf1d)
        return psf1d # wl x depth x size x size

    def to_sensor_phase(self):
        # x_prime = self.mask_pitch * torch.arange(1, self.mask_size + 1).reshape(-1, 1, 1, 1)
        # y_prime = self.mask_pitch * torch.arange(1, self.mask_size + 1).reshape(1, -1, 1, 1)
        x = self.camera_pixel_pitch * torch.arange(1, self.image_size[0] + 1).reshape(-1, 1)
        y = self.camera_pixel_pitch * torch.arange(1, self.image_size[1] + 1).reshape(1, -1)
        # print(x_prime.shape)
        # print(x.shape)
        # distance_diff = (x - x_prime) ** 2 + (y - y_prime) **2
        distance_diff = x ** 2 + y ** 2
        distance_diff = distance_diff.unsqueeze(0).to(self.device)
        print( distance_diff.shape)
        z = self.sensor_distance()
        k = (2 * math.pi) / self.wavelengths.reshape(-1, 1, 1).double()
        return k * distance_diff / (2 * z)
    
    def __reshaping(self, tensor):
        temp = tensor.size(2) * tensor.size(3)
        tensor = tensor.reshape(tensor.size(0), tensor.size(1), -1, tensor.size(4), tensor.size(5))
        tensor = tensor.reshape(tensor.size(0), tensor.size(1), temp, -1)
        return tensor

    # def _psf_at_camera_impl(self, H, rho_grid, rho_sampling, ind, size, modulate_phase):
    #     # As this quadruple will be copied to the other three, rho = 0 is avoided.
    #     psf1d = self.psf1d(H, modulate_p
    # hase)
    #     # psf1d: n_wl X D X n_rho
    #     # rho_grid: n_wl X n_rho
    #     # rho_sampling: n_wl X (n_rho - 2) X (n_rho - 2)
    #     # ind: n_wl X (n_rho - 2) X (n_rho - 2)
    #     psf_rd = F.relu(cubicspline.interp(rho_grid, psf1d, rho_sampling, ind).float())
    #     psf_rd = psf_rd.reshape(self.n_wl, self.n_depths, size[0] // 2, size[1] // 2)
    #     return utils.helper.copy_quadruple(psf_rd)    