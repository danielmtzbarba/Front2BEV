""" 
MIT License

Copyright (c) 2019 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

# ------------------------------------------------------------------------------------
import torch.nn as nn
import torchvision.models as models

from src.models.nn.ved_blocks import *
# ------------------------------------------------------------------------------------

class VED(nn.Module):

    def __init__(self, k_classes=4):
        super(VED, self).__init__()

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