import torch
import torch.nn as nn
from collections import namedtuple
OutputsContainer = namedtuple('OutputContainer', field_names=['est_images', 'est_depthmaps'])

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer, momentum=0.01):
        super().__init__()
        if norm_layer is nn.Identity:
            bias = True
        else:
            bias = False

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=bias),
            norm_layer(out_ch, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=bias),
            norm_layer(out_ch, momentum=momentum),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class DownsampleBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.block = ConvBlock(in_ch, out_ch, norm_layer=norm_layer)
        self.downsample = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.block(x)
        y = x
        x = self.downsample(x)
        return x, y


class UpsampleBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.block = ConvBlock(in_ch, out_ch, norm_layer=norm_layer)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, y):
        x = self.upsample(x)
        x = torch.cat([x, y], dim=1)
        x = self.block(x)
        return x


class UNet(nn.Module):

    def __init__(self, channels, n_layers: int, norm_layer=None):
        super().__init__()

        self.downblocks = nn.ModuleList()
        self.upblocks = nn.ModuleList()
        self.n_layers = n_layers

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        for i in range(n_layers):
            block = DownsampleBlock(channels[i], channels[i + 1], norm_layer)
            self.downblocks.append(block)

        bottom_in = channels[n_layers]
        bottom_out = channels[n_layers + 1]
        self.bottom_block = ConvBlock(bottom_in, bottom_out, norm_layer=norm_layer)

        for i in range(n_layers):
            block = UpsampleBlock(channels[i + 1] + channels[i + 2], channels[i + 1], norm_layer)
            self.upblocks.append(block)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = []
        for i in range(self.n_layers):
            x, y = self.downblocks[i](x)
            features.append(y)
        x = self.bottom_block(x)
        for i in range(self.n_layers - 1, -1, -1):
            x = self.upblocks[i](x, features[i])
        return x
    

class CNN(nn.Module):

    def __init__(self, hparams, *args, **kargs):
        super().__init__()
        self.preinverse = hparams.preinverse
        depth_ch = 1
        color_ch = 3
        rgba_ch = 4
        n_layers = 4
        n_depths = hparams.n_depths
        base_ch = hparams.model_base_ch
        preinv_input_ch = color_ch * n_depths + color_ch
        base_input_layers = nn.Sequential(
            nn.Conv2d(preinv_input_ch, preinv_input_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(preinv_input_ch),
            nn.ReLU(),
            nn.Conv2d(preinv_input_ch, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
        )
        # Without the preinverse input, it has ((color_ch * preinv_input_ch) + preinv_input_ch * 2) more parameters than
        # with the preinverse input. (255 params)
        if self.preinverse:
            input_layers = base_input_layers
        else:
            input_layers = nn.Sequential(
                nn.Conv2d(color_ch, preinv_input_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(preinv_input_ch),
                nn.ReLU(),
                base_input_layers,
            )

        output_layers = nn.Sequential(
            nn.Conv2d(base_ch, color_ch + depth_ch, kernel_size=1, bias=True)
        )
        self.decoder = nn.Sequential(
            input_layers,
            UNet(
                channels=[base_ch, base_ch, 2 * base_ch, 2 * base_ch, 4 * base_ch, 4 * base_ch],
                n_layers=n_layers,
            ),
            output_layers,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, captimgs, pinv_volumes, *args, **kargs):
        b_sz, c_sz, d_sz, h_sz, w_sz = pinv_volumes.shape
        if self.preinverse:
            inputs = torch.cat([captimgs.unsqueeze(2), pinv_volumes], dim=2)
        else:
            inputs = captimgs.unsqueeze(2)
        est = torch.sigmoid(self.decoder(inputs.reshape(b_sz, -1, h_sz, w_sz)))
        est_images = est[:, :-1]
        est_depthmaps = est[:, [-1]]
        outputs = OutputsContainer(
            est_images=est_images,
            est_depthmaps=est_depthmaps,
        )
        return outputs
