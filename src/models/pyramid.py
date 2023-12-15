"""
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
 To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/.
"""

import torch.nn as nn

class PyramidOccupancyNetwork(nn.Module):

    def __init__(self, frontend, transformer, topdown, classifier):
        super().__init__()

        self.frontend = frontend
        self.transformer = transformer
        self.topdown = topdown
        self.classifier = classifier
    

    def forward(self, image, calib, *args):

        # Extract multiscale feature maps
        feature_maps = self.frontend(image)

        # Transform image features to birds-eye-view
        bev_feats = self.transformer(feature_maps, calib)

        # Apply topdown network
        td_feats = self.topdown(bev_feats)

        # Predict individual class log-probabilities
        logits = self.classifier(td_feats)
        return logits