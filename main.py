import numpy as np
import argparse
import os
import tqdm
import cv2

import torch
from torch.utils.data import DataLoader

from utils import Set_seed, cal_loss, visualize_landmark
from dataset import Gaze_dataset
from model import EyeNet

import warnings
warnings.filterwarnings("ignore")

torch.autograd.set_detect_anomaly(True)
def valid(args, model, valid_loader, epoch):
    model.eval()
    pbar = tqdm.tqdm(total=len(valid_loader), ncols=0, desc="Valid[%d/%d]"%(epoch, args.n_epochs), unit=" step")

    heatmap_loss_total = 0
    landmark_loss_total = 0
    gaze_loss_total = 0
    total_loss = 0
    with torch.no_grad():
        val_losses = []
        for idx, (image, label, heatmap_gt) in enumerate(valid_loader):
            image = image.float().cuda()
            heatmap_gt = heatmap_gt.cuda()
            gaze = label[:,0,:].cuda() # (pitch, yaw)
            landmark = label[:,1:,:].cuda() # iris, lid, pupil

            heatmaps_pred, landmarks_pred, gaze_pred = model(image)

            heatmaps_loss, landmarks_loss, gaze_loss = cal_loss(heatmaps_pred, heatmap_gt, landmarks_pred, landmark, gaze_pred, gaze, args.nstack)
            loss = 1000 * heatmaps_loss + landmarks_loss + 1000 * gaze_loss

            total_loss += loss.data
            heatmap_loss_total += heatmaps_loss.data
            landmark_loss_total += landmarks_loss.data
            gaze_loss_total += gaze_loss.data

            pbar.update()
            pbar.set_postfix(
            heatmap_loss = f"{heatmap_loss_total:.4f}",
            landmarks_loss = f"{landmark_loss_total:.4f}",
            gaze_loss = f"{gaze_loss_total:.4f}",
            total_loss = f"{total_loss / idx:.4f}"
            )
            val_losses.append(loss.item())

        visualize_landmark(args, image, landmarks_pred, landmark, heatmap_gt, heatmaps_pred, os.path.join(args.saved_pred, 'valid'))
        val_loss = np.mean(val_losses)
        return val_loss

def train(args, epoch, model, train_loader, optimizer):
    pbar = tqdm.tqdm(total=len(train_loader), ncols=0, desc="Train[%d/%d]"%(epoch, args.n_epochs), unit=" step")
    model.train()

    # statistics
    heatmap_loss_total = 0
    landmark_loss_total = 0
    gaze_loss_total = 0
    total_loss = 0
    for idx, (image, label, heatmap_gt) in enumerate(train_loader):
        image = image.cuda()
        heatmap_gt = heatmap_gt.cuda() # landmark heatmaps
        gaze = label[:,0,:].cuda() # (pitch, yaw)
        landmark = label[:,1:,:].cuda() # lid, pupil

        optimizer.zero_grad()
        heatmaps_pred, landmarks_pred, gaze_pred = model(image)

        heatmaps_loss, landmarks_loss, gaze_loss = cal_loss(heatmaps_pred, heatmap_gt, landmarks_pred, landmark, gaze_pred, gaze, args.nstack)

        loss = 1000 * heatmaps_loss + landmarks_loss + 1000 * gaze_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.data
        heatmap_loss_total += heatmaps_loss.data
        landmark_loss_total += landmarks_loss.data
        gaze_loss_total += gaze_loss.data

        pbar.update()
        pbar.set_postfix(
        heatmap_loss = f"{heatmap_loss_total:.4f}",
        landmarks_loss = f"{landmark_loss_total:.4f}",
        gaze_loss = f"{gaze_loss_total:.4f}",
        total_loss = f"{total_loss / idx:.4f}"
        )

        if idx % 100 == 0:
            visualize_landmark(args, image, landmarks_pred, landmark, heatmap_gt, heatmaps_pred, os.path.join(args.saved_pred, 'train'))
            
    pbar.close()

def main(args, model, train_loader, valid_loader, optimizer):

    save_name = os.path.join(args.saved_model, 'model_best.pth')
    print('start tarining...')

    min_loss = 100000000000000000
    for epoch in range(args.n_epochs):
        _ = train(args, epoch, model, train_loader, optimizer)
        valid_loss = valid(args, model, valid_loader, epoch)

        if valid_loss <= min_loss:
            min_loss = valid_loss
            torch.save({'model': model.state_dict()}, save_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ''' Paths '''
    # parser.add_argument('--data_dir', type=str, default="/home/brianw0924/hdd/TEyeD")
    parser.add_argument('--data_dir', type=str, default="../dataset")
    parser.add_argument('--dataset', type=str, default="TEyeD", choices=["Neurobit", "TEyeD"]) # if you set Neurobit, will do random cropping
    parser.add_argument('--saved_model', type=str, default='./checkpoints/gaze_landmark')
    parser.add_argument('--saved_pred', type=str, default='./test_image')
    parser.add_argument('--load', type=str, default='')
    
    """model parameters"""
    parser.add_argument('--nstack', type=int, default=3)
    parser.add_argument('--nfeatures', type=int, default=64, help='Number of feature maps to use.')
    parser.add_argument('--nlandmarks', type=int, default=25, help='Number of landmarks to be predicted.')

    ''' paramters '''
    parser.add_argument('--image_width', type=int, default=192, help='Image width')
    parser.add_argument('--image_height', type=int, default=144, help='Image height')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    args = parser.parse_args()
    os.makedirs(args.saved_model, exist_ok=True)
    os.makedirs(os.path.join(args.saved_pred, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.saved_pred, 'valid'), exist_ok=True)
    Set_seed(args.seed)

    train_data = Gaze_dataset(args, 'train')
    valid_data = Gaze_dataset(args, 'valid')

    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False, shuffle=False)
    print('Number of training data : ', len(train_data))
    print('Number of validation data : ', len(valid_data))
    
    model = EyeNet(args, nstack=args.nstack, nfeatures=args.nfeatures, nlandmarks=args.nlandmarks).cuda()
    if args.load:
        model.load_state_dict(torch.load(args.load))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    main(args, model, train_loader, valid_loader, optimizer)