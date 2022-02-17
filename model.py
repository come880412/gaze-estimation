import torch
from torch import nn
from model_component import Conv, Hourglass, Pool, Residual

import numpy as np

# To fine the maximize value of the heatmap
def softargmax2d(input, beta=100, dtype=torch.float32):
    *_, h, w = input.shape

    input = input.reshape(*_, h * w)
    input = nn.functional.softmax(beta * input, dim=-1)

    indices_c, indices_r = np.meshgrid(
        np.linspace(0, 1, w),
        np.linspace(0, 1, h),
        indexing='xy'
    )

    indices_r = torch.tensor(np.reshape(indices_r, (-1, h * w)))
    indices_c = torch.tensor(np.reshape(indices_c, (-1, h * w)))

    device = input.get_device()
    if device >= 0:
        indices_r = indices_r.to(device)
        indices_c = indices_c.to(device)

    result_r = torch.sum((h - 1) * input * indices_r, dim=-1)
    result_c = torch.sum((w - 1) * input * indices_c, dim=-1)

    result = torch.stack([result_r, result_c], dim=-1)

    return result.type(dtype)

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)


class EyeNet(nn.Module):
    def __init__(self, args, nstack, nfeatures, nlandmarks, bn=False, increase=0, **kwargs):
        super(EyeNet, self).__init__()

        self.img_w = args.image_width
        self.img_h = args.image_height
        self.nstack = nstack
        self.nfeatures = nfeatures
        self.nlandmarks = nlandmarks
        self.nstack = nstack

        self.heatmap_w = self.img_w / 2
        self.heatmap_h = self.img_h / 2

        self.nstack = nstack
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 1, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, nfeatures)
        )

        self.pre2 = nn.Sequential(
            Conv(nfeatures, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, nfeatures)
        )

        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass(4, nfeatures, bn, increase),
            ) for i in range(nstack)])

        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(nfeatures, nfeatures),
                Conv(nfeatures, nfeatures, 1, bn=True, relu=True)
            ) for i in range(nstack)])

        self.outs = nn.ModuleList([Conv(nfeatures, nlandmarks, 1, relu=False, bn=False) for i in range(nstack)])
        self.merge_features = nn.ModuleList([Merge(nfeatures, nfeatures) for i in range(nstack - 1)])
        self.merge_preds = nn.ModuleList([Merge(nlandmarks, nfeatures) for i in range(nstack - 1)])

        self.gaze_out = nn.Sequential(
            nn.Linear(in_features=int(nfeatures * self.img_w * self.img_h / 64 + nlandmarks*2), out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=2)
        )

    def forward(self, imgs):
        # imgs of size 1,ih,iw
        x = self.pre(imgs)

        gaze_x = self.pre2(x)
        gaze_x = gaze_x.flatten(start_dim=1)

        combined_hm_preds = []
        for i in torch.arange(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)

        heatmaps_out = torch.stack(combined_hm_preds, 1)

        # preds = N x nlandmarks * heatmap_h * heatmap_w
        landmarks_out = softargmax2d(preds)  # N x nlandmarks x 2

        # Gaze
        gaze = torch.cat((gaze_x, landmarks_out.flatten(start_dim=1)), dim=1)
        gaze = self.gaze_out(gaze)

        return heatmaps_out, landmarks_out, gaze