import random
from re import L
from tkinter import Scale
import numpy as np
import cv2

import torch
import torch.nn as nn


def Set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def vec2angle(gaze):
    '''
    gaze shape : (x, y, z)
    
    * from your viewpoint *
    x+: right
    y+: down
    z+: point to you
    pitch+: up
    yaw+: left(your pov) ; right (patient's pov)
    '''
    x, y, z = gaze[0], gaze[1], gaze[2]
    pitch = - np.arctan(y/z) * 180 / np.pi
    yaw = - np.arctan(x/z) * 180 / np.pi
    return np.array([yaw, pitch])[np.newaxis,:]

def gaussian_2d(w, h, cx, cy, sigma=1.0):
    """Generate heatmap with single 2D gaussian."""
    xs, ys = np.meshgrid(
        np.linspace(0, w - 1, w, dtype=np.float32),
        np.linspace(0, h - 1, h, dtype=np.float32)
    )

    assert xs.shape == (h, w)
    alpha = -0.5 / (sigma ** 2)
    heatmap = np.exp(alpha * ((xs - cx) ** 2 + (ys - cy) ** 2))
    return heatmap


def get_heatmaps(w, h, landmarks):
    heatmaps = []
    for (y, x) in landmarks:
        heatmaps.append(gaussian_2d(w, h, cx=x, cy=y, sigma=2.0))
    return np.array(heatmaps)

class HeatmapLoss(torch.nn.Module):
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        loss = ((pred - gt)**2)
        loss = torch.mean(loss, dim=(1, 2, 3))
        return loss

def cal_loss(combined_hm_preds, heatmaps, landmarks_pred, landmarks, gaze_pred, gaze, n_stack):
    heatmapLoss = HeatmapLoss()
    landmarks_loss = nn.MSELoss()
    gaze_loss = nn.MSELoss()

    combined_loss = []
    for i in range(n_stack):
        combined_loss.append(heatmapLoss(combined_hm_preds[:, i, :], heatmaps))

    heatmap_loss = torch.stack(combined_loss, dim=1)
    landmarks_loss = landmarks_loss(landmarks_pred, landmarks)
    gaze_loss = gaze_loss(gaze_pred, gaze)

    return torch.sum(heatmap_loss), landmarks_loss, gaze_loss

def visualize_landmark(args, image, landmarks_pred, landmark, heatmap_gt, heatmaps_pred, save_path):
    image = image.cpu().detach().numpy()[-1]
    image = image.transpose((1,2,0))

    out_w, out_h = args.image_width, args.image_height
    _, _, heatmap_h, heatmap_w = heatmap_gt.shape
    scale_w, scale_h = out_w / heatmap_w, out_h / heatmap_h

    landmarks_pred = landmarks_pred.cpu().detach().numpy()[-1]
    landmark_image_pred = image.copy()
    for l in landmarks_pred:
        y, x = l
        landmark_image_pred = cv2.circle(landmark_image_pred, (int(round(x * scale_w)),int(round(y * scale_h))), 1, (0,0,255), 3)
    
    landmark = landmark.cpu().detach().numpy()[-1]
    landmark_image_true = image.copy()
    for l in landmark:
        y, x = l
        landmark_image_true = cv2.circle(landmark_image_true, (int(round(x * scale_w)),int(round(y * scale_h))), 1, (0,0,255), 3)

    hm = np.mean(heatmap_gt[-1, :].cpu().detach().numpy(), axis=0)
    hm_pred = np.mean(heatmaps_pred[-1, -1, :].cpu().detach().numpy(), axis=0)
    norm_hm = cv2.normalize(hm, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_hm_pred = cv2.normalize(hm_pred, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    cv2.imwrite('./%s/true_heatmap.jpg' % save_path, norm_hm * 255)
    cv2.imwrite('./%s/pred_heatmap.jpg' % save_path, norm_hm_pred * 255)
    cv2.imwrite('./%s/eye.jpg' % save_path, image * 255)
    cv2.imwrite('./%s/pred_eye_landmark.jpg' % save_path, landmark_image_pred * 255)
    cv2.imwrite('./%s/true_eye_landmark.jpg' % save_path, landmark_image_true * 255)
