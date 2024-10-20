import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
#from optics.camera import Camera, RotationallySymmetricCamera, AsymmetricMaskRotationallySymmetricCamera
import argparse
import os
from datetime import datetime
import imageio
from datasets.dualpixel import DualPixel
from datasets.sceneflow import SceneFlow
from models.model2 import DepthEstimator
import utils.helper
from PIL import Image
import numpy as np
import utils.IO as IO
from utils.helper import crop_boundary, gray_to_rgb, imresize, linear_to_srgb, srgb_to_linear, to_bayer
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional.regression import mean_absolute_error, mean_squared_error
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
# global variable
# date = datetime.now().strftime("%Y%m%d_%H%M%S")
# tensorboard_path = 'exp' + str(date)
# writer = SummaryWriter(os.path.join('runs', tensorboard_path))
global_step = 0

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

    # resolution
    parser.add_argument('--mask_sz', type=int, default=8000)
    parser.add_argument('--image_sz', type=int, default=256) # sensor resolution (crop batch)
    parser.add_argument('--full_size', type=int, default=1920)
    # physical length (meter)
    parser.add_argument('--sensor_diameter', type=float, default=2.4768e-3)
    parser.add_argument('--focal_length', type=float, default=50e-3)
    parser.add_argument('--focal_depth', type=float, default=1.7) #d
    parser.add_argument('--f_number', type=float, default=6.3) #use to 


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

    train_subset_size = 400
    sf_train_dataset = SceneFlow('train',
                                 (image_sz + 4 * crop_width,
                                  image_sz + 4 * crop_width),
                                 is_training=True,
                                 randcrop=randcrop, augment=augment, padding=padding,
                                 singleplane=False)
    train_subset_indices = torch.randperm(len(sf_train_dataset))[:train_subset_size]
    sf_train_dataset = torch.utils.data.Subset(sf_train_dataset,train_subset_indices)

    val_subset_size = 100
    sf_val_dataset = SceneFlow('val',
                               (image_sz + 4 * crop_width,
                                image_sz + 4 * crop_width),
                               is_training=False,
                               randcrop=randcrop, augment=augment, padding=padding,
                               singleplane=False)
    val_subset_indices = torch.randperm(len(sf_val_dataset))[:val_subset_size]
    sf_val_dataset = torch.utils.data.Subset(sf_val_dataset, val_subset_indices)

    if hparams.mix_dualpixel_dataset:
        train_subset_size2 = 300
        dp_train_dataset = DualPixel('train',
                                     (image_sz + 4 * crop_width,
                                      image_sz + 4 * crop_width),
                                     is_training=True,
                                     randcrop=randcrop, augment=augment, padding=padding)
        train_subset_indices2 = torch.randperm(len(dp_train_dataset))[:train_subset_size2]
        dp_train_dataset = torch.utils.data.Subset(dp_train_dataset, train_subset_indices2)

        val_subset_size2 = 60
        dp_val_dataset = DualPixel('val',
                                   (image_sz + 4 * crop_width,
                                    image_sz + 4 * crop_width),
                                   is_training=False,
                                   randcrop=randcrop, augment=augment, padding=padding)
        val_subset_indices2 = torch.randperm(len(dp_val_dataset))[:val_subset_size2]
        dp_val_dataset = torch.utils.data.Subset(dp_val_dataset, val_subset_indices2)

        train_dataset = torch.utils.data.ConcatDataset([dp_train_dataset, sf_train_dataset])
        val_dataset = torch.utils.data.ConcatDataset([dp_val_dataset, sf_val_dataset])

        n_sf = len(sf_train_dataset)
        n_dp = len(dp_train_dataset)
        sample_weights = torch.cat([1. / n_dp * torch.ones(n_dp, dtype=torch.double),
                                    1. / n_sf * torch.ones(n_sf, dtype=torch.double)], dim=0)
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams.batch_sz, sampler=sampler,
                                      num_workers=hparams.num_workers, shuffle=False, pin_memory=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=hparams.batch_sz,
                                    num_workers=hparams.num_workers, shuffle=False, pin_memory=True)
    else:
        train_dataset = sf_train_dataset
        val_dataset = sf_val_dataset
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams.batch_sz,
                                      num_workers=hparams.num_workers, shuffle=True, pin_memory=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=hparams.batch_sz,
                                    num_workers=hparams.num_workers, shuffle=False, pin_memory=True)

    return train_dataloader, val_dataloader

def training_step(model, samples, batch_idx, device):
        target_images = samples['image'].to(device)
        target_depthmaps = samples['depthmap'].to(device)
        depth_conf = samples['depth_conf'].to(device)

        if depth_conf.ndim == 4:
            depth_conf = crop_boundary(depth_conf, model.crop_width * 2)

        outputs = model(target_images, target_depthmaps)

        # Unpack outputs
        est_images = outputs.est_images
        est_depthmaps = outputs.est_depthmaps
        target_images = outputs.target_images
        target_depthmaps = outputs.target_depthmaps
        captimgs_linear = outputs.captimgs_linear

        data_loss, loss_logs = model.compute_loss(outputs, target_depthmaps, target_images, depth_conf)
        #loss_logs = {f'train_loss/{key}': val for key, val in loss_logs.items()}
        loss_logs = {f'{key}': val for key, val in loss_logs.items()}
        return data_loss, loss_logs

def validation_step(model, samples, batch_idx, device):
        target_images = samples['image'].to(device)
        target_depthmaps = samples['depthmap'].to(device)
        depth_conf = samples['depth_conf'].to(device)
        if depth_conf.ndim == 4:
            depth_conf = crop_boundary(depth_conf, 2 * model.crop_width)

        outputs = model(target_images, target_depthmaps)

        # Unpack outputs
        est_images = outputs.est_images
        est_depthmaps = outputs.est_depthmaps
        target_images = outputs.target_images
        target_depthmaps = outputs.target_depthmaps

        est_depthmaps = est_depthmaps * depth_conf
        target_depthmaps = target_depthmaps * depth_conf

        mae_depthmap = mean_absolute_error(est_depthmaps, target_depthmaps)
        mse_depthmap = mean_squared_error(est_depthmaps, target_depthmaps)
        mae_image = mean_absolute_error(est_images, target_images)
        #must add .contiguous() since it demand for .view()
        mse_image = mean_squared_error(est_images.contiguous(), target_images.contiguous())
        psnr_image = peak_signal_noise_ratio(est_images, target_images)
        ssim_image = structural_similarity_index_measure(est_images, target_images)
        vgg_image = model.image_lossfn.train_loss(est_images, target_images)       
        val_losses = {
            'mae_depthmap' : mae_depthmap,
            'mse_depthmap' : mse_depthmap,
            'mae_image' : mae_image,
            'mse_image' : mse_image,
            'psnr_image' : psnr_image, 
            'ssim_image' : ssim_image,
            'vgg_image' : vgg_image
        }
        return val_losses

def validation_epoch_end(model, outputs1, outputs2):
    # do something with the predictions from all validation_steps
    mae_depthmap = torch.mean(torch.tensor(outputs1))
    vgg_image = torch.mean(torch.tensor(outputs2))
    val_loss = model.combine_loss(mae_depthmap, vgg_image, 0.)
    return val_loss

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define the colormap for color mapping
    # cmap = LinearSegmentedColormap.from_list('custom', [(0, 'red'), (0.5, 'green'), (1, 'blue')])
    cmap = LinearSegmentedColormap.from_list('custom', [(0, 'black'), (1, 'white')])
    # Normalize the tensor values to the range [0, 1]
    parser = arg_parser()
    hparams = parser.parse_args()
    train_data_loader, val_data_loader = prepare_data(hparams)
    model = DepthEstimator(hparams).to(device)
    optimizer = model.configure_optimizers()
    epoch = 0
    writer = SummaryWriter(os.path.join('data','runs', 'exp' + datetime.now().strftime("%Y%m%d_%H%M%S")))
    if hparams.checkpoint:
        checkpoint = torch.load(hparams.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        global_step = checkpoint['global_step']
 
    while epoch < hparams.max_epochs:
        optics_lr = model.hparams.optics_lr
        cnn_lr = model.hparams.cnn_lr
        model.train()
        for batch_idx, batch in enumerate(train_data_loader):
            loss, train_log = training_step(model, batch, batch_idx, device)
            # clear gradients
            optimizer.zero_grad()
            # backward
            loss.backward()
            # update parameters
            # if global_step < 4000:
            #     lr_scale = min(1., float(global_step + 1) / 4000.)
            #     optimizer.param_groups[0]['lr'] = lr_scale * optics_lr
            #     optimizer.param_groups[1]['lr'] = lr_scale * cnn_lr
            optimizer.step()
            writer.add_scalars('train_loss', train_log, global_step)
            global_step = global_step + 1
        model.eval()
        with torch.no_grad():
            mae_depthmap_outs = []
            vgg_image_outs = []
            mae_depthmap = mse_depthmap = mae_image = mse_image = psnr_image = ssim_image = vgg_image = 0.0
            for batch_idx, batch in enumerate(val_data_loader):
                val_losses = validation_step(model, batch, batch_idx, device) 
                mae_depthmap += val_losses['mae_depthmap']
                mse_depthmap += val_losses['mse_depthmap']
                mae_image += val_losses['mae_image']
                mse_image += val_losses['mse_image']
                psnr_image += val_losses['psnr_image']
                ssim_image += val_losses['ssim_image']
                vgg_image += val_losses['vgg_image']
                mae_depthmap_outs.append(mae_depthmap)
                vgg_image_outs.append(vgg_image)
            val_loss = validation_epoch_end(model, mae_depthmap_outs, vgg_image_outs)
            scaler = 1 / len(val_data_loader)
            mae_depthmap *= scaler
            mse_depthmap *= scaler
            mae_image *= scaler
            mse_image *= scaler
            psnr_image *= scaler
            ssim_image *= scaler
            vgg_image *= scaler
            writer.add_scalars('val_loss', {
                    'mae_depthmap' : mae_depthmap,
                    'mse_depthmap' : mse_depthmap, 
                    'mae_image' : mae_image,
                    'mse_image' : mse_image,
                    'psnr_image' : psnr_image,
                    'ssim_image' : ssim_image, 
                    'vgg_image' : vgg_image,
                    'val_loss' : val_loss
            }, epoch)
        #save checkpoint
        path = os.path.join('data', 'checkpoints') 
        if not os.path.exists(path):
            os.makedirs(path)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(path, f'model_epoch{epoch}_loss{loss:.4f}_valloss{val_loss:.4f}_{current_time}.pt') 
        #with open(path, 'w') as file:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'global_step': global_step,
            }, path)
        print(f'Checkpoint saved: {path} (loss: {loss:.4f}) (val_loss: {val_loss:.4f})')
        epoch += 1
    print("global_step: ")
    print(global_step)
    print("\nEND.")

