import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from optics.camera import Camera, RotationallySymmetricCamera, AsymmetricMaskRotationallySymmetricCamera
import argparse
import os
import imageio
def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train a differentiable camera')

    # 
    parser.add_argument('--experiment_name', type=str, default='LearnedDepth')
    parser.add_argument('--mix_dualpixel_dataset', dest='mix_dual_pixel_dataset', action='store_true')
    parser.set_defaults(mix_dualpixel_dataset=True)

    # logger parameters
    parser.add_argument('--summary_max_images', type=int, default=4)
    parser.add_argument('--summary_image_sz', type=int, default=256)
    parser.add_argument('--summary_mask_sz', type=int, default=256)
    parser.add_argument('--summary_depth_every', type=int, default=1)
    parser.add_argument('--summary_track_train_every', type=int, default=4000)

    # training parameters
    parser.add_argument('--cnn_lr', type=float, default=1e-3)
    parser.add_argument('--optics_lr', type=float, default=1e-9)
    parser.add_argument('--batch_sz', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--randcrop', default=False, action='store_true')
    parser.add_argument('--augment', default=False, action='store_true')

    # loss parameters
    parser.add_argument('--depth_loss_weight', type=float, default=1.0)
    parser.add_argument('--image_loss_weight', type=float, default=1.0)
    parser.add_argument('--psf_loss_weight', type=float, default=1.0)
    parser.add_argument('--psf_size', type=int, default=64)

    # dataset parameters
    parser.add_argument('--image_sz', type=int, default=256)
    parser.add_argument('--n_depths', type=int, default=16)
    parser.add_argument('--min_depth', type=float, default=1.0)
    parser.add_argument('--max_depth', type=float, default=5.0)
    parser.add_argument('--crop_width', type=int, default=32)

    # solver parameters
    parser.add_argument('--reg_tikhonov', type=float, default=1.0)
    parser.add_argument('--model_base_ch', type=int, default=32)

    parser.add_argument('--preinverse', dest='preinverse', action='store_true')
    parser.add_argument('--no-preinverse', dest='preinverse', action='store_false')
    parser.set_defaults(preinverse=True)

    # optics parameters
    parser.add_argument('--camera_type', type=str, default='mixed')
    parser.add_argument('--mask_sz', type=int, default=2048)
    #parser.add_argument('--focal_length', type=float, default=50e-3)
    parser.add_argument('--focal_length', type=float, default=100e-3)
    parser.add_argument('--focal_depth', type=float, default=1.7)
    #parser.add_argument('--focal_depth', type=float, default= float('inf'))
    parser.add_argument('--f_number', type=float, default=6.3)
    parser.add_argument('--camera_pixel_pitch', type=float, default=6.45e-6)
    parser.add_argument('--noise_sigma_min', type=float, default=0.001)
    parser.add_argument('--noise_sigma_max', type=float, default=0.005)
    parser.add_argument('--full_size', type=int, default=1920)
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
        default_root_dir='data/logs',
        max_epochs=100,
    )

    return parser

def dump_images(results: dict, output_dir, cmap):
    for title, image in results.items():
        plt.figure()
        # normalized term
        image = image.cpu()
        if(title == 'psf'):
            min_value = image.min().item()
            max_value = image.max().item()
            image = (image - min_value) / (max_value - min_value)
        # normalized term
        plt.imshow(image, cmap=cmap)
        plt.title(title)
        #plt.axis('off')  # Hide axes
        plt.colorbar()
        # Define the filename for saving
        filename = os.path.join(output_dir, f'{title}.png')
        # Save the image as a file
        # plt.savefig(filename)
        # plt.close()
        imageio.imwrite(filename, image)


# Define the colormap for color mapping
# cmap = LinearSegmentedColormap.from_list('custom', [(0, 'red'), (0.5, 'green'), (1, 'blue')])
cmap = LinearSegmentedColormap.from_list('custom', [(0, 'black'), (1, 'white')])

# Normalize the tensor values to the range [0, 1]
parser = arg_parser()
hparams = parser.parse_args()
mask_diameter = hparams.focal_length / hparams.f_number
wavelengths = [632e-9, 550e-9, 450e-9]
camera_recipe = {
            'wavelengths': wavelengths,
            'min_depth': hparams.min_depth,
            'max_depth': hparams.max_depth,
            'focal_depth': hparams.focal_depth,
            'n_depths': hparams.n_depths,
            'image_size': hparams.image_sz + 4 * hparams.crop_width,
            'camera_pixel_pitch': hparams.camera_pixel_pitch,
            'focal_length': hparams.focal_length,
            'mask_diameter': mask_diameter,
            'mask_size': hparams.mask_sz,
        }
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
camera = AsymmetricMaskRotationallySymmetricCamera(**camera_recipe, requires_grad=False).to(device)
dump_images(camera.dump(), './results', cmap)


# tensor = (camera.get_psf()[0, 0]).cpu()
# min_value = tensor.min().item()
# max_value = tensor.max().item()
# normalized_tensor = (tensor - min_value) / (max_value - min_value)
# print(normalized_tensor)
# print("sensor_distance: ")
# print(camera.sensor_distance())
# # Plot the tensor with the specified colormap
# plt.imshow(normalized_tensor, cmap=cmap)
# plt.title('psf')
# plt.colorbar()

# # Show the plot
# plt.show()