# Vendored from https://github.com/princeton-vl/WAFT (BSD-3-Clause)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_gaussian(size, sigma=1.0):
    x = torch.arange(size, dtype=torch.float32) - size // 2
    y = torch.arange(size, dtype=torch.float32) - size // 2
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    d = xx ** 2 + yy ** 2
    g = torch.exp(-d / (2 * sigma ** 2))
    return g


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class InferenceWrapper:
    def __init__(self, model, scale=0.0, train_size=None, pad_to_train_size=False, tiling=False):
        self.model = model
        self.scale = scale
        self.train_size = train_size
        self.pad_to_train_size = pad_to_train_size
        self.tiling = tiling

    @torch.no_grad()
    def calc_flow(self, image1, image2, iters=None):
        if self.scale > 0:
            h, w = image1.shape[-2:]
            new_h = int(h * self.scale)
            new_w = int(w * self.scale)
            image1 = F.interpolate(image1, (new_h, new_w), mode='bilinear', align_corners=True)
            image2 = F.interpolate(image2, (new_h, new_w), mode='bilinear', align_corners=True)
        output = self.model(image1, image2, iters=iters)
        if self.scale > 0:
            for i in range(len(output['flow'])):
                output['flow'][i] = F.interpolate(output['flow'][i], (h, w), mode='bilinear', align_corners=True)
                output['flow'][i] = output['flow'][i] / self.scale
        return output
