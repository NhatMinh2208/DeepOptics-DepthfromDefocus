import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from optics.camera import Camera, RotationallySymmetricCamera, AsymmetricMaskRotationallySymmetricCamera
import argparse
import os
import imageio
from datasets.dualpixel import DualPixel
from datasets.sceneflow import SceneFlow
from models.model import DepthEstimator
import utils.helper
from PIL import Image

import numpy as np
import utils.IO as IO
def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train a differentiable camera')

    # 
    parser.add_argument('--experiment_name', type=str, default='LearnedDepth')
    parser.add_argument('--mix_dualpixel_dataset', dest='mix_dual_pixel_dataset', action='store_true')
    #parser.set_defaults(mix_dualpixel_dataset=True)
    parser.set_defaults(mix_dualpixel_dataset=False)
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
    parser.add_argument('--mask_sz', type=int, default=384)
    parser.add_argument('--image_sz', type=int, default=256) # sensor resolution
    # parser.add_argument('--mask_sz', type=int, default=400+128)
    # parser.add_argument('--image_sz', type=int, default=400) # sensor resolution
    parser.add_argument('--full_size', type=int, default=1920) # extended sensor resolution (not used currently)

    # physical length (meter)
    #parser.add_argument('--mask_diameter', type=float, default=2.4768e-3)
    # parser.add_argument('--mask_diameter', type=float, default=2.4768e-3)
    parser.add_argument('--mask_diameter', type=float, default=5e-3)
    parser.add_argument('--sensor_diameter', type=float, default=2.4768e-3)
    #parser.add_argument('--focal_length', type=float, default=50e-3)
    parser.add_argument('--focal_length', type=float, default=100e-3) #f
    # parser.add_argument('--focal_depth', type=float, default=1.7)
    # parser.add_argument('--focal_depth', type=float, default=1.7442)
    parser.add_argument('--focal_depth', type=float, default=5.) #d
    # parser.add_argument('--focal_depth', type=float, default=1000000000.0)
    #parser.add_argument('--focal_depth', type=float, default= float('inf'))
    parser.add_argument('--f_number', type=float, default=6.3) #use to 
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
        default_root_dir='data/logs',
        max_epochs=100,
    )

    return parser

def dump_images(results: dict, output_dir, cmap):
    for title, image in results.items():
        #plt.figure()
        # normalized term
        image = image.cpu()
        if(title == 'psf'):
            min_value = image.min().item()
            max_value = image.max().item()
            image = (image - min_value) / (max_value - min_value)
        # normalized term
        # plt.imshow(image, cmap=cmap)
        # plt.title(title)
        # #plt.axis('off')  # Hide axes
        # plt.colorbar()
        # Define the filename for saving
        filename = os.path.join(output_dir, f'{title}.png')
        # Save the image as a file
        # plt.savefig(filename)
        # plt.close()
        imageio.imwrite(filename, image)

def dump_images2(results: dict, output_dir):
    for title, image in results.items():
        n_channel = image.shape[0]
        filename = os.path.join(output_dir, f'{title}.png')
        if (n_channel == 3):
            image_np = image.cpu().numpy().transpose((1, 2, 0))
            # Normalize the values back to the [0, 255] range (assuming the tensor values were normalized)
            image_np = ((image_np - image_np.min()) / (image_np.max() - image_np.min()) * 255).astype('uint8')
            # Create a PIL image from the NumPy array
            image_pil = Image.fromarray(image_np)
            # Save the image using PIL
            directory = os.path.dirname(filename)
            if not os.path.exists(directory):
                os.makedirs(directory)
            image_pil.save(filename)
        if (n_channel == 1):
            image = image.squeeze(0)
            image_np = image.cpu().numpy()
            image_pil = Image.fromarray(image_np.astype('uint8'), mode='L')  # 'L' mode for single-channel (grayscale)
            directory = os.path.dirname(filename)
            if not os.path.exists(directory):
                os.makedirs(directory)
            # Save the image
            image_pil.save(filename)

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
    
    if hparams.mix_dualpixel_dataset:
        dp_train_dataset = DualPixel('train',
                                     (image_sz + 4 * crop_width,
                                      image_sz + 4 * crop_width),
                                     is_training=True,
                                     randcrop=randcrop, augment=augment, padding=padding)

        train_dataset = torch.utils.data.ConcatDataset([dp_train_dataset, sf_train_dataset])
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
    else:
        train_dataset = sf_train_dataset
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams.batch_sz,
                                      num_workers=hparams.num_workers, shuffle=True, pin_memory=True)
    return train_dataloader

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define the colormap for color mapping
    # cmap = LinearSegmentedColormap.from_list('custom', [(0, 'red'), (0.5, 'green'), (1, 'blue')])
    cmap = LinearSegmentedColormap.from_list('custom', [(0, 'black'), (1, 'white')])
    # Normalize the tensor values to the range [0, 1]
    parser = arg_parser()
    hparams = parser.parse_args()
    mask_diameter = hparams.focal_length / hparams.f_number
    #mask_diameter = 0.0024768
    wavelengths = [632e-9, 550e-9, 450e-9]
    real_image_size = hparams.image_sz + 4 * hparams.crop_width
    camera_recipe = {
                'wavelengths': wavelengths,
                'min_depth': hparams.min_depth,
                'max_depth': hparams.max_depth,
                'focal_depth': hparams.focal_depth,
                'n_depths': hparams.n_depths,
                'image_size': hparams.image_sz + 4 * hparams.crop_width,
                'camera_pixel_pitch': hparams.sensor_diameter / real_image_size, #hparams.camera_pixel_pitch,
                #'camera_pixel_pitch': hparams.camera_pixel_pitch,
                'focal_length': hparams.focal_length,
                'mask_diameter': hparams.mask_diameter,
                #'mask_diameter': mask_diameter,
                'mask_size': hparams.mask_sz,
            }
    camera = AsymmetricMaskRotationallySymmetricCamera(**camera_recipe, requires_grad=False).to(device)
    # #dumping thing
    
    #dump_images(camera.dump(), './results', cmap)
    dump_images(camera.dump_depth_psf(), './depth_psf_results', cmap)
    # testing with real image
    
    #data_loader = prepare_data(args)
    model = DepthEstimator(hparams).to(device)
    # optimizer = model.configure_optimizers()

    data_loader = prepare_data(hparams)
    train_example = next(iter(data_loader))
    target_images = train_example['image'].to(device)
    target_depthmaps = train_example['depthmap'].to(device)
    depth_conf = train_example['depth_conf'].to(device)

    # target_images = torch.rand([1, 3, 384, 384]).to(device)
    # target_depthmaps =  torch.rand([1, 1, 384, 384]).to(device)
    # depth_conf = torch.ones([1, 1, 384, 384]).to(device)

    result = model(target_images, target_depthmaps)

    # print(target_images.shape) #torch.Size([1, 3, 256, 256]) 
    # print(target_depthmaps.shape) #torch.Size([1, 1, 384, 384])
    # print(depth_conf.shape) #torch.Size([1, 1, 384, 384])
    # print(result.est_images.shape) #torch.Size([1, 3, 256, 256])
    # print(result.captimgs.shape) #torch.Size([1, 3, 256, 256]
    dictionary = {}
    # ['captimgs', 'captimgs_linear',
    #                                       'est_images', 'est_depthmaps',
    #                                       'target_images', 'target_depthmaps',
    #                                       'psf'])

    #1106 281223
    #---------------------------------------
    dictionary['target_image'] = target_images[0]
    dictionary['target_depthmap'] = target_depthmaps[0]
    dictionary['result_captimgs'] = result.captimgs[0]
    dump_images2(dictionary, './model_results')

    print(target_depthmaps[0][0]) 
    ips_depth_map = utils.helper.ips_to_metric(target_depthmaps[0][0].cpu(), hparams.min_depth, hparams.max_depth)
    #plt.imshow(target_depthmaps[0][0].cpu(), cmap=cmap)
    plt.imshow(ips_depth_map, cmap=cmap)
    plt.title('depth_map')
    plt.colorbar()
    # Show the plot
    plt.show()
    #---------------------------------------




    #show depth map

#     DATA_ROOT = os.path.join('data', 'training_data', 'SceneFlow')
#     disparity_dir = os.path.join(DATA_ROOT, 'FlyingThings3D_subset', 'train', 'disparity', 'right') 
#     image_dir = os.path.join(DATA_ROOT, 'FlyingThings3D_subset', 'train', 'image_clean', 'right')
    
#     #disparity = np.flip(IO.read(os.path.join(disparity_dir, 'A_00000006.pfm')), axis=0).astype(np.float32)
#     disparity = (IO.read(os.path.join(disparity_dir, 'A_00000006.pfm')).astype(np.float32))
#     disparity -= disparity.min()
#     disparity /= disparity.max()
#     disparity = 1. - disparity
#     # max_val = disparity.max()
#     # disparity = max_val - disparity
#     disparity_full = disparity

#     #disparity = disparity[150:150+384, 550:550+384]
#     disparity = disparity[0:528, 120:120+528]
#     print(disparity.shape)
#     plt.figure()
#     ips_depth_map = utils.helper.ips_to_metric(disparity, hparams.min_depth, hparams.max_depth)
#     plt.imshow(ips_depth_map, cmap=cmap)
#    # plt.imshow(disparity, cmap=cmap)
#     plt.title('depth_map')
#     plt.colorbar()

#     # # Show the plot
#     #plt.show()

#     image_np = imageio.imread(os.path.join(image_dir, 'A_00000006.png')).astype(np.float32)
#     #image_np = ((image_np - image_np.min()) / (image_np.max() - image_np.min()) * 255).astype('uint8')
#     #image_np = image_np[150:150+384, 550:550+384, :]
#     image_np = image_np[0:528, 120:120+528, :]
#     image_np /= 255
#     print(image_np.shape)
#     # plt.figure()
#     # plt.imshow(image_np)
#     # plt.title('image')
#     # Show the plot
#     #plt.show()
#     image_np = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)
#     disparity = torch.from_numpy(disparity).unsqueeze(0).unsqueeze(1).to(device)
#     print(image_np.shape)
#     print(disparity.shape)
#     result = model(image_np, disparity)
#     dictionary['target_image'] = image_np[0]
#     dictionary['target_depthmap'] = disparity[0]
#     dictionary['result_captimgs'] = result.captimgs[0]
#     dump_images2(dictionary, './model_results')
#     plt.show()




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