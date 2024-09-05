'''
- Choose 2 camera types you want to compare
- Visualize the results of 2 cameras, including: psf, captured images,...
- Visualize the differences between 2 cameras
'''

import torch
import argparse
import os

from datasets.dualpixel import DualPixel
from datasets.sceneflow import SceneFlow

from optics.camera7 import Camera, RotationallySymmetricCamera, AsymmetricMaskRotationallySymmetricCamera, MixedCamera
def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Compare different cameras')
    # Specify two camera to compare 
    #parser.add_argument('--camera1', type=str, default='mixed')
    parser.add_argument('--camera', type=str, default='asym')

    # Using both dual pixel and scene flow dataset ?
    parser.add_argument('--mix_dualpixel_dataset', dest='mix_dual_pixel_dataset', action='store_true')
    parser.set_defaults(mix_dualpixel_dataset=True)

    # logger parameters
    parser.add_argument('--summary_max_images', type=int, default=4)
    parser.add_argument('--summary_image_sz', type=int, default=256)
    parser.add_argument('--summary_mask_sz', type=int, default=256)
    parser.add_argument('--summary_depth_every', type=int, default=1)

    # dataset-related parameters
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--randcrop', default=False, action='store_true')
    parser.add_argument('--augment', default=False, action='store_true')

    parser.add_argument('--n_depths', type=int, default=16)
    parser.add_argument('--min_depth', type=float, default=1.0)
    parser.add_argument('--max_depth', type=float, default=5.0)
    parser.add_argument('--crop_width', type=int, default=32)

    # camera(s) parameters
    #1024
    parser.add_argument('--mask_sz', type=int, default=800) # whatever put in front of sensor (mask) resolution
    parser.add_argument('--image_sz', type=int, default=256) # sensor resolution (crop batch)
    parser.add_argument('--full_size', type=int, default=600) # real sensor resolution 
    # physical length (meter)
    parser.add_argument('--mask_diameter', type=float, default=2.5e-3)
    parser.add_argument('--sensor_diameter', type=float, default=2.4768e-3)
    parser.add_argument('--focal_length', type=float, default=50e-3) # f
    parser.add_argument('--focal_depth', type=float, default=1.7) # d
    
    parser.add_argument('--f_number', type=float, default=20) #use to 
    #parser.add_argument('--camera_pixel_pitch', type=float, default=6.45e-6)

    # noise_level
    parser.add_argument('--noise_sigma_min', type=float, default=0.001)
    parser.add_argument('--noise_sigma_max', type=float, default=0.005)
    parser.add_argument('--mask_upsample_factor', type=int, default=1)
    parser.add_argument('--diffraction_efficiency', type=float, default=0.7)

    parser.add_argument('--bayer', dest='bayer', action='store_true')
    parser.add_argument('--no-bayer', dest='bayer', action='store_false')
    parser.set_defaults(bayer=True)
    parser.add_argument('--occlusion', dest='occlusion', action='store_true')
    parser.add_argument('--no-occlusion', dest='occlusion', action='store_false')
    parser.set_defaults(occlusion=True)
    parser.add_argument('--optimize_optics', dest='optimize_optics', action='store_true')
    parser.add_argument('--no-optimize_optics', dest='optimize_optics', action='store_false')
    parser.set_defaults(optimize_optics=False)

    # model parameters
    parser.add_argument('--psfjitter', dest='psf_jitter', action='store_true')
    parser.add_argument('--no-psfjitter', dest='psf_jitter', action='store_false')
    parser.set_defaults(psf_jitter=True)

    parser.set_defaults(
        gpus=1,
        default_root_dir='result_logs',
        # max_epochs=100,
        # max_epochs=2,
    )
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None, help='load checkpoint')
    return parser

def prepare_data(hparams):
    image_sz = hparams.image_sz
    crop_width = hparams.crop_width
    augment = hparams.augment
    randcrop = hparams.randcrop
    padding = 0

    subset_size = 400
    sf_dataset = SceneFlow('train',
                                 (image_sz + 4 * crop_width,
                                  image_sz + 4 * crop_width),
                                 is_training=True,
                                 randcrop=randcrop, augment=augment, padding=padding,
                                 singleplane=False)
    subset_indices = torch.randperm(len(sf_dataset))[:subset_size]
    sf_dataset = torch.utils.data.Subset(sf_dataset,subset_indices)

    if hparams.mix_dualpixel_dataset:
        subset_size2 = 300
        dp_dataset = DualPixel('train',
                                     (image_sz + 4 * crop_width,
                                      image_sz + 4 * crop_width),
                                     is_training=True,
                                     randcrop=randcrop, augment=augment, padding=padding)
        subset_indices2 = torch.randperm(len(dp_dataset))[:subset_size2]
        dp_dataset = torch.utils.data.Subset(dp_dataset, subset_indices2)

        dataset = torch.utils.data.ConcatDataset([dp_dataset, sf_dataset])
       
        n_sf = len(sf_dataset)
        n_dp = len(dp_dataset)
        sample_weights = torch.cat([1. / n_dp * torch.ones(n_dp, dtype=torch.double),
                                    1. / n_sf * torch.ones(n_sf, dtype=torch.double)], dim=0)
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=hparams.batch_sz, sampler=sampler,
                                      num_workers=hparams.num_workers, shuffle=False, pin_memory=True)
    else:
        dataset = sf_dataset
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=hparams.batch_sz,
                                      num_workers=hparams.num_workers, shuffle=True, pin_memory=True)
    return dataloader

import os
from PIL import Image
import matplotlib.pyplot as plt

def visualize_images(input_folder, output_folder, filename):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    if not os.path.exists(input_path):
        print(f"Input image {input_path} does not exist.")
        return
    
    if not os.path.exists(output_path):
        print(f"Output image {output_path} does not exist.")
        return

    # Load images
    input_image = Image.open(input_path)
    output_image = Image.open(output_path)

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Display input image
    axes[0].imshow(input_image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    # Display output image
    axes[1].imshow(output_image)
    axes[1].set_title('Output Image')
    axes[1].axis('off')

    # Show the plot
    plt.show()

    # # Example usage
    # input_folder = 'path/to/input_folder'
    # output_folder = 'path/to/output_folder'
    # filename = 'example_image.png'
    # visualize_images(input_folder, output_folder, filename)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define the colormap for color mapping
    # cmap = LinearSegmentedColormap.from_list('custom', [(0, 'red'), (0.5, 'green'), (1, 'blue')])
    #cmap = LinearSegmentedColormap.from_list('custom', [(0, 'black'), (1, 'white')])
    # Normalize the tensor values to the range [0, 1]
    parser = arg_parser()
    hparams = parser.parse_args()
    #data_loader = prepare_data(hparams)
   
    mask_diameter = hparams.focal_length / hparams.f_number
    wavelengths = [632e-9, 550e-9, 450e-9]
    real_image_size = hparams.image_sz + 4 * hparams.crop_width
    camera_recipe = {
        'wavelengths': wavelengths,
        'min_depth': hparams.min_depth,
        'max_depth': hparams.max_depth,
        'focal_depth': hparams.focal_depth,
        'n_depths': hparams.n_depths,
        'image_size': hparams.image_sz + 4 * hparams.crop_width,
        'camera_pixel_pitch': hparams.sensor_diameter / real_image_size,
        'focal_length': hparams.focal_length,
        #'mask_diameter': hparams.mask_diameter,
        'mask_diameter': mask_diameter,
        'mask_size': hparams.mask_sz,
        'debug': hparams.debug
    }
    optimize_optics = hparams.optimize_optics

    camera_recipe['mask_upsample_factor'] = hparams.mask_upsample_factor
    camera_recipe['diffraction_efficiency'] = hparams.diffraction_efficiency
    camera_recipe['full_size'] = hparams.full_size
    #camera = MixedCamera(**camera_recipe, requires_grad=optimize_optics)
    camera = AsymmetricMaskRotationallySymmetricCamera(**camera_recipe, requires_grad=optimize_optics)
    #print(camera.heightmap2().shape)
    psf = camera.get_psf()[0][0].cpu()
    # #psf = camera.heightmap()
    # # psf = camera.dump_conv_real[0][8].cpu()
    plt.imshow(psf, cmap='gray')
    plt.colorbar()  # Optional: add a color bar to show the intensity scale
    plt.title('Grayscale Image')
    plt.axis('off')  # Optional: turn off the axis
    plt.show()

    # psf = camera.dump_conv[0][8].cpu()
    # plt.imshow(psf, cmap='gray')
    # plt.colorbar()  # Optional: add a color bar to show the intensity scale
    # plt.title('Grayscale Image')
    # plt.axis('off')  # Optional: turn off the axis
    # plt.show()
    # if hparams.camera1 == 'mixed':
    #     camera1 = MixedCamera(**camera_recipe, requires_grad=optimize_optics)
    # elif hparams.camera1 == 'asym':
    #     camera1 = AsymmetricMaskRotationallySymmetricCamera(**camera_recipe, requires_grad=optimize_optics)
    # else:
    #     raise Exception("Undefined camera type!")
    
    # if hparams.camera2 == 'mixed':
    #     camera2 = MixedCamera(**camera_recipe, requires_grad=optimize_optics)
    # elif hparams.camera2 == 'asym':
    #     camera2 = AsymmetricMaskRotationallySymmetricCamera(**camera_recipe, requires_grad=optimize_optics)
    # else:
    #     raise Exception("Undefined camera type!")    
 
    

