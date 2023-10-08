import torch
import torch.nn as nn
import torchvision.models as models

# ------------------------------------------------------------------------------------
import numpy as np

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()
# ------------------------------------------------------------------------------------

class upsample(nn.Module):

    def __init__(self, if_deconv, channels=None):
        super(upsample, self).__init__()
        if if_deconv:
            self.upsample = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1, bias=False)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.upsample(x)

        return x

class double_conv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class encoder_after_vgg(nn.Module):

    def __init__(self):
        super(encoder_after_vgg, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.mu_dec = nn.Linear(4096, 512)
        self.logvar_dec = nn.Linear(4096, 512)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 4096)
        mu = self.mu_dec(x)
        logvar = self.logvar_dec(x)

        return mu, logvar


class decoder_conv(nn.Module):
    def __init__(self, if_deconv, k_classes=4):
        super(decoder_conv, self).__init__()

        self.up1 = upsample(if_deconv=if_deconv, channels=128)
        self.conv1 = double_conv(128, 256)
        self.up2 = upsample(if_deconv=if_deconv, channels=256)
        self.conv2 = double_conv(256, 256)
        self.up3 = upsample(if_deconv=if_deconv, channels=256)
        self.conv3 = double_conv(256, 256)
        self.up4 = upsample(if_deconv=if_deconv, channels=256)
        self.conv4 = double_conv(256, 256)
        self.up5 = upsample(if_deconv=if_deconv, channels=256)
        self.conv5 = double_conv(256, 256)
        self.conv_out = nn.Conv2d(256, k_classes, 3, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        x = x.view(-1, 128, 2, 2)
        x = self.up1(x)
        x = self.conv1(x)

        x = self.up2(x)
        x = self.conv2(x)

        x = self.up3(x)
        x = self.conv3(x)

        x = self.up4(x)
        x = self.conv4(x)

        x = self.up5(x)
        x = self.conv5(x)

        x = self.conv_out(x)

        return x
    
# ------------------------------------------------------------------------------------
class VAE(nn.Module):

    def __init__(self, k_classes=4):
        super(VAE, self).__init__()

        self.vgg16 = models.vgg16_bn(weights=True)
        self.vgg16_feature = nn.Sequential(*list(self.vgg16.features.children())[:])
        self.encoder_afterv_vgg = encoder_after_vgg()
        self.decoder = decoder_conv(if_deconv=True, k_classes=k_classes)

    def reparameterize(self, is_training, mu, logvar):
        if is_training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, is_training, defined_mu=None):
        x = self.vgg16_feature(x)
        mu, logvar = self.encoder_afterv_vgg(x)
        z = self.reparameterize(is_training, mu, logvar)
        if defined_mu is not None:
            z = defined_mu
        pred_map = self.decoder(z)

        return pred_map, mu, logvar
# ------------------------------------------------------------------------------------
import torch.optim as optim

def get_vae_train_objs(n_classes):
    model = VAE(n_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)
    
    return {'model': model,
            'optimizer': optimizer,
            'scheduler':scheduler
            }