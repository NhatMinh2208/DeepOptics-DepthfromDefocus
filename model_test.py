from datasets.dualpixel import DualPixel
from datasets.sceneflow import SceneFlow
import torch.utils.data
import argparse
import os
from models.model import DepthEstimator
from torch.utils.tensorboard import SummaryWriter
from utils.helper import crop_boundary, gray_to_rgb, imresize, linear_to_srgb, srgb_to_linear, to_bayer


writer = SummaryWriter(comment='_' + 'train')

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
    parser.add_argument('--focal_length', type=float, default=50e-3)
    parser.add_argument('--focal_depth', type=float, default=1.7)
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

def prepare_data(hparams):
    image_sz = hparams.image_sz
    crop_width = hparams.crop_width
    augment = hparams.augment
    randcrop = hparams.randcrop
    padding = 0
    val_idx = 3994
    sf_train_dataset = SceneFlow('train',
                                 (image_sz + 4 * crop_width,
                                  image_sz + 4 * crop_width),
                                 is_training=True,
                                 randcrop=randcrop, augment=augment, padding=padding,
                                 singleplane=False)
    
    #
    #sf_train_dataset = torch.utils.data.Subset(sf_train_dataset,
    #                                           range(val_idx, len(sf_train_dataset)))

    # sf_val_dataset = SceneFlow('train',
    #                            (image_sz + 4 * crop_width,
    #                             image_sz + 4 * crop_width),
    #                            is_training=False,
    #                            randcrop=randcrop, augment=augment, padding=padding,
    #                            singleplane=False)
    # sf_val_dataset = torch.utils.data.Subset(sf_val_dataset, range(val_idx))
    if hparams.mix_dualpixel_dataset:
        dp_train_dataset = DualPixel('train',
                                     (image_sz + 4 * crop_width,
                                      image_sz + 4 * crop_width),
                                     is_training=True,
                                     randcrop=randcrop, augment=augment, padding=padding)
        # dp_val_dataset = DualPixel('val',
        #                            (image_sz + 4 * crop_width,
        #                             image_sz + 4 * crop_width),
        #                            is_training=False,
        #                            randcrop=randcrop, augment=augment, padding=padding)

        train_dataset = torch.utils.data.ConcatDataset([dp_train_dataset, sf_train_dataset])
        #val_dataset = torch.utils.data.ConcatDataset([dp_val_dataset, sf_val_dataset])

        n_sf = len(sf_train_dataset)
        n_dp = len(dp_train_dataset)
        print("n_sf=",n_sf)
        print("n_dp=", n_dp)
        sample_weights = torch.cat([1. / n_dp * torch.ones(n_dp, dtype=torch.double),
                                    1. / n_sf * torch.ones(n_sf, dtype=torch.double)], dim=0)
        #Samples elements from [0,..,len(weights)-1] with given probabilities (weights)
        #Used for sample from  different size of datasets
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams.batch_sz, sampler=sampler,
                                      num_workers=hparams.num_workers, shuffle=False, pin_memory=True)
       # val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=hparams.batch_sz,
       #                             num_workers=hparams.num_workers, shuffle=False, pin_memory=True)
    else:
        train_dataset = sf_train_dataset
        #val_dataset = sf_val_dataset
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams.batch_sz,
                                      num_workers=hparams.num_workers, shuffle=True, pin_memory=True)
        # val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=hparams.batch_sz,
        #                             num_workers=hparams.num_workers, shuffle=False, pin_memory=True)

    return train_dataloader

def training_step(model, device, data_loader, optimizer, args):
    for train_batch in data_loader:
        target_images = train_batch['image'].to(device)
        target_depthmaps = train_batch['depthmap'].to(device)
        depth_conf = train_batch['depth_conf'].to(device)

        if depth_conf.ndim == 4:
            depth_conf = crop_boundary(depth_conf, args.crop_width * 2)
        outputs = model(target_images, target_depthmaps)
        # Unpack outputs
        est_images = outputs.est_images
        est_depthmaps = outputs.est_depthmaps
        target_images = outputs.target_images
        target_depthmaps = outputs.target_depthmaps
        captimgs_linear = outputs.captimgs_linear

        data_loss, loss_logs = model.compute_loss(outputs, target_depthmaps, target_images, depth_conf)

        #logging things
        loss_logs = {f'{key}': val for key, val in loss_logs.items()}
        misc_logs = {
            'target_depth_max': target_depthmaps.max(),
            'target_depth_min': target_depthmaps.min(),
            'est_depth_max': est_depthmaps.max(),
            'est_depth_min': est_depthmaps.min(),
            'target_image_max': target_images.max(),
            'target_image_min': target_images.min(),
            'est_image_max': est_images.max(),
            'est_image_min': est_images.min(),
            'captimg_max': captimgs_linear.max(),
            'captimg_min': captimgs_linear.min(),
        }
        # if args.optimize_optics:
        #     misc_logs.update({
        #         'optics/heightmap_max': model.camera.heightmap1d().max(),
        #         'optics/heightmap_min': model.camera.heightmap1d().min(),
        #         'optics/psf_out_of_fov_energy': loss_logs['train_loss/psf_loss'],
        #         'optics/psf_out_of_fov_max': loss_logs['train_loss/psf_out_of_fov_max'],
        #     })
        # logs = {}
        # logs.update(loss_logs)
        # logs.update(misc_logs)
        #model.log_dict(logs)
        writer.add_scalars('train_loss', loss_logs, epoch)
        writer.add_scalars('train_misc', misc_logs, epoch)
        #optimize
        data_loss.backward()
        #model.optimizer_step()
        # warm up lr
        # if self.trainer.global_step < 4000:
        #     lr_scale = min(1., float(self.trainer.global_step + 1) / 4000.)
        #     optimizer.param_groups[0]['lr'] = lr_scale * args.optics_lr
        #     optimizer.param_groups[1]['lr'] = lr_scale * args.cnn_lr
        # update params
        optimizer.step()
        optimizer.zero_grad()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'Num GPUs: {torch.cuda.device_count()}')

    # parse command line
    parser = arg_parser()
    args = parser.parse_args()
    #data_loader = prepare_data(args)

    model = DepthEstimator(args).to(device)
    optimizer = model.configure_optimizers()

    target_images = torch.rand([1, 3, 384, 384]).to(device)
    target_depthmaps =  torch.rand([1, 1, 384, 384]).to(device)
    depth_conf = torch.ones([1, 1, 384, 384]).to(device)
    result = model(target_images, target_depthmaps)
    print(result)
    # for train_batch in data_loader:
    #     target_images = train_batch['image'].to(device)
    #     target_depthmaps = train_batch['depthmap'].to(device)
    #     depth_conf = train_batch['depth_conf'].to(device)
    #     result = model(target_images, target_depthmaps)
    #     print(result)
    #     print("image: ")
    #     print(target_images)
    #     print("depth: ")
    #     print(target_depthmaps) 
    #     print("conf: ")
    #     print(depth_conf) 
    # for epoch in range(5):
    #     training_step(model, device, data_loader, optimizer, args)