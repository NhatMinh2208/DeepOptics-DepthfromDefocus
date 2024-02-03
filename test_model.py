"""
Usage

python run_trained_snapshotdepth_on_captured_images.py \
    --scene indoor --captimg_path data/captured_data/indoor2_predemosaic.tif \
    --ckpt_path data/checkpoints/checkpoint.ckpt
"""

import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import torch

from models.model2 import DepthEstimator
from optics.image_reconstruction import apply_tikhonov_inverse
from utils.fft import crop_psf
from utils.helper import crop_boundary, linear_to_srgb

def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train a differentiable camera')

    # 
    parser.add_argument('--experiment_name', type=str, default='LearnedDepth')
    parser.add_argument('--mix_dualpixel_dataset', dest='mix_dual_pixel_dataset', action='store_true')
    parser.set_defaults(mix_dualpixel_dataset=True)
    #parser.set_defaults(mix_dualpixel_dataset=False)
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
    #parser.add_argument('--n_depths', type=int, default=16)
    parser.add_argument('--n_depths', type=int, default=16)
    #parser.add_argument('--min_depth', type=float, default=1.0)
    #parser.add_argument('--min_depth', type=float, default= 0.001)
    #parser.add_argument('--max_depth', type=float, default=5.0)
    #parser.add_argument('--max_depth', type=float, default=1.0)
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
    # parser.add_argument('--mask_sz', type=int, default=2048)

    # resolution
    #parser.add_argument('--mask_sz', type=int, default=384) # whatever put in front of sensor (mask) resolution
    parser.add_argument('--mask_sz', type=int, default=8000)
    parser.add_argument('--image_sz', type=int, default=256) # sensor resolution (crop batch)
    # parser.add_argument('--mask_sz', type=int, default=400+128)
    # parser.add_argument('--image_sz', type=int, default=400) # sensor resolution
    #parser.add_argument('--full_size', type=int, default=600) # real sensor resolution 
    parser.add_argument('--full_size', type=int, default=1920)
    # physical length (meter)
    #parser.add_argument('--mask_diameter', type=float, default=2.4768e-3)
    # parser.add_argument('--mask_diameter', type=float, default=2.4768e-3)
    #parser.add_argument('--mask_diameter', type=float, default=2.5e-3)
    parser.add_argument('--sensor_diameter', type=float, default=2.4768e-3)
    #parser.add_argument('--focal_length', type=float, default=50e-3)
    #parser.add_argument('--focal_length', type=float, default=50e-3) #f
    parser.add_argument('--focal_length', type=float, default=50e-3)
    # parser.add_argument('--focal_depth', type=float, default=1.7)
    # parser.add_argument('--focal_depth', type=float, default=1.7442)
    parser.add_argument('--focal_depth', type=float, default=1.7) #d
    # parser.add_argument('--focal_depth', type=float, default=1000000000.0)
    #parser.add_argument('--focal_depth', type=float, default= float('inf'))
    parser.add_argument('--f_number', type=float, default=20) #use to 
    #parser.add_argument('--f_number', type=float, default=6.3) #use to 
    #parser.add_argument('--camera_pixel_pitch', type=float, default=6.45e-6)


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

def to_uint8(x: torch.Tensor):
    """
    x: B x C x H x W
    """
    return (255 * x.squeeze(0).clamp(0, 1)).permute(1, 2, 0).to(torch.uint8)


def strech_img(x):
    return (x - x.min()) / (x.max() - x.min())


def find_minmax(img, saturation=0.1):
    min_val = np.percentile(img, saturation)
    max_val = np.percentile(img, 100 - saturation)
    return min_val, max_val


def rescale_image(x):
    min, max = find_minmax(x)
    return (x - min) / (max - min)


def average_inference(x):
    x = torch.stack([
        x[0],
        torch.flip(x[1], dims=(-1,)),
        torch.flip(x[2], dims=(-2,)),
        torch.flip(x[3], dims=(-2, -1)),
    ], dim=0)
    return x.mean(dim=0, keepdim=True)


@torch.no_grad()
def main(args):
    device = torch.device('cpu')

    # Load the saved checkpoint
    # This is not a default way to load the checkpoint through Lightning.
    # My code cleanup made it difficult to directly load the checkpoint from what I used for the paper.
    # So, manually loading the learnable parameters to the model.
    ckpt = torch.load(args.ckpt_path, map_location=lambda storage, loc: storage)
    model = DepthEstimator(hparams=args)

    model.camera.heightmap1d_.data = ckpt['model_state_dict']['camera.heightmap1d_']
    print("lens height map: ", model.camera.heightmap1d_.data.size(), model.camera.heightmap1d_.data)

    decoder_dict = {key[8:]: value for key, value in ckpt['model_state_dict'].items() if 'decoder' in key}
    model.decoder.load_state_dict(decoder_dict)
    model.eval()

    save_name = os.path.splitext(os.path.basename(args.captimg_path))[0]
    captimg_linear = torch.from_numpy(skimage.io.imread(args.captimg_path).astype(np.float32)).unsqueeze(0)

    # Remove the offset value of the camera
    captimg_linear -= 64

    # add batch dim
    captimg_linear = captimg_linear.unsqueeze(0)

    # Debayer with the bilinear interpolation
    captimg_linear = model.debayer(captimg_linear)

    # Adjust white balance (The values are estimated from a white paper and manually tuned.)
    if 'indoor1' in save_name:
        captimg_linear[:, 0] *= (40293.078 - 64) / (34013.722 - 64) * 1.03
        captimg_linear[:, 2] *= (40293.078 - 64) / (13823.391 - 64) * 0.97
    elif 'indoor2' in save_name:
        captimg_linear[:, 0] *= (38563. - 64) / (28537. - 64) * 0.94
        captimg_linear[:, 2] *= (38563. - 64) / (15134. - 64) * 1.13
    elif 'outdoor' in save_name:
        captimg_linear[:, 0] *= (61528.274 - 64) / (46357.955 - 64) * 0.9
        captimg_linear[:, 2] *= (61528.274 - 64) / (36019.744 - 64) * 1.4
    else:
        raise ValueError('white balance is not set.')

    captimg_linear /= captimg_linear.max()

    # Inference-time augmentation
    captimg_linear = torch.cat([
        captimg_linear,
        torch.flip(captimg_linear, dims=(-1,)),
        torch.flip(captimg_linear, dims=(-2,)),
        torch.flip(captimg_linear, dims=(-1, -2)),
    ], dim=0)

    image_sz = captimg_linear.shape[-2:]

    captimg_linear = captimg_linear.to(device)
    model = model.to(device)

    psf = model.camera.normalize_psf(model.camera.psf_at_camera(size=image_sz).unsqueeze(0))
    psf_cropped = crop_psf(psf, image_sz)
    pinv_volumes = apply_tikhonov_inverse(captimg_linear, psf_cropped, model.hparams.reg_tikhonov,
                                          apply_edgetaper=True)
    model_outputs = model.decoder(captimgs=captimg_linear, pinv_volumes=pinv_volumes)

    est_images = crop_boundary(model_outputs.est_images, model.crop_width)
    est_depthmaps = crop_boundary(model_outputs.est_depthmaps, model.crop_width)
    capt_images = linear_to_srgb(crop_boundary(captimg_linear[[0]], model.crop_width))

    est_images = average_inference(est_images)
    est_depthmaps = average_inference(est_depthmaps)

    # Save the results
    skimage.io.imsave(f'data/result/{save_name}_captimg.png', to_uint8(rescale_image(capt_images)))
    skimage.io.imsave(f'data/result/{save_name}_estimg.png', to_uint8(rescale_image(est_images)))
    plt.imsave(f'data/result/{save_name}_estdepthmap.png',
               (255 * (1 - est_depthmaps).squeeze().clamp(0, 1)).to(torch.uint8), cmap='inferno')


if __name__ == '__main__':
    parser = arg_parser()
    parser.add_argument('--captimg_path', type=str,  default='/captured_data/indoor1_predemosaic.tiff')
    parser.add_argument('--ckpt_path', type=str, default='data/checkpoints/model_epoch11_loss0.2139_20240202_191604.pt')
    args = parser.parse_args()
    main(args)
