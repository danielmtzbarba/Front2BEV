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

"""
This implementation of the model from the paper "Cross-view Semantic 
Segmentation for Sensing Surroundings" is directly adapted from the code
provided by the original authors at
https://github.com/pbw-Berwin/View-Parsing-Network (accessed 08/06/2020)

"""
# ------------------------------------------------------------------------------------
import torch.nn as nn

from src.models.nn.vpn_blocks import resnet18, TransformModule, PPMBilinear
# ------------------------------------------------------------------------------------


class VPNModel(nn.Module):
    def __init__(self, num_views, num_class, output_size, fc_dim, map_extents, 
                 map_resolution):

        super(VPNModel, self).__init__()
        self.num_views = num_views
        self.output_size = output_size

        self.seg_size = (
            int((map_extents[3] - map_extents[1]) / map_resolution),
            int((map_extents[2] - map_extents[0]) / map_resolution),
        )

        # MODIFIED: we fix the transform module, the encoder and decoder to be 
        # the ones described in the paper 
        self.encoder = resnet18(True)
        self.transform_module = TransformModule(dim=self.output_size, 
                                                num_view=self.num_views)
        self.decoder = PPMBilinear(num_class, fc_dim, False)

    def forward(self, x, *args):
        B, N, C, H, W = x.view([-1, self.num_views, int(x.size()[1] / self.num_views)] \
                            + list(x.size()[2:])).size()

        x = x.view(B*N, C, H, W)
        x = self.encoder(x)[0]
        x = x.view([B, N] + list(x.size()[1:]))
        x = self.transform_module(x)
        x = self.decoder([x], self.seg_size)
        
        return x