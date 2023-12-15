"""
Create a visualization of a front2bev sequence of images .

"""

import torch
import numpy as np

from dan.utils.torch import get_torch_device, load_model

class Front2BEV(object):
    def __init__(self,
                 model,
                 ckpt_path = None,
                 device = get_torch_device()):
        
        self._model = model
        self._device = device

        self._setup_model(ckpt_path)
    
    def _setup_model(self, ckpt_path):
        self._model = load_model(self._model, ckpt_path)
        self._model = self._model.to(self._device)
        self._model.eval()

    def transform(self, front_img):
        front_img = front_img.unsqueeze(0).float().to(self._device)

        with torch.inference_mode():
            logits, _, _ = self._model(front_img, False)
            logits = logits.cpu().numpy().transpose((0, 2, 3, 1))
            pred_map = np.argmax(logits, axis=3)

        return np.reshape(pred_map ,[64, 64]).astype(np.uint8)