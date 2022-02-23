import numpy as np
import argparse
import os
import tqdm

import torch
from torch.utils.data import DataLoader

from utils import Set_seed, cal_loss, visualize_landmark
from dataset import Dikablis_data
from model import EyeNet

import warnings
warnings.filterwarnings("ignore")

def valid(args, model, valid_loader, step):
    model.eval()
    pbar = tqdm.tqdm(total=len(valid_loader), ncols=0, desc="Valid[%d/%d]"%(step, args.total_steps), unit=" step")

    heatmap_loss_total = 0
    landmark_loss_total = 0
    gaze_loss_total = 0
    total_loss = 0
    with torch.no_grad():
        val_losses = []
        for idx, (image, label, heatmap_gt) in enumerate(valid_loader):
            image = image.float().cuda()
            heatmap_gt = heatmap_gt.cuda()
            gaze = label[:,0,:].cuda() # (yaw, pitch)
            landmark = label[:,1:,:].cuda() # lid, pupil

            heatmaps_pred, landmarks_pred, gaze_pred = model(image)

            heatmaps_loss, landmarks_loss, gaze_loss = cal_loss(heatmaps_pred, heatmap_gt, landmarks_pred, landmark, gaze_pred, gaze, args.nstack)
            loss = 100 * heatmaps_loss + 50* landmarks_loss + 500 * gaze_loss

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
        pbar.close()
        val_loss = np.mean(val_losses)
        return val_loss

def train(args, image, label, heatmap_gt, model, optimizer, step):
    model.train()

    image = image.cuda()
    heatmap_gt = heatmap_gt.cuda() # landmark heatmaps
    gaze = label[:,0,:].cuda() # (yaw, pitch)
    landmark = label[:,1:,:].cuda() # lid, pupil

    optimizer.zero_grad()
    heatmaps_pred, landmarks_pred, gaze_pred = model(image)

    heatmaps_loss, landmark_loss, gaze_loss = cal_loss(heatmaps_pred, heatmap_gt, landmarks_pred, landmark, gaze_pred, gaze, args.nstack)

    loss = 100 * heatmaps_loss + 50* landmark_loss + 500 * gaze_loss

    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        visualize_landmark(args, image, landmarks_pred, landmark, heatmap_gt, heatmaps_pred, os.path.join(args.saved_pred, 'train'))
            
    return loss, gaze_loss, heatmaps_loss, landmark_loss

def main(args, model, train_loader, valid_loader, optimizer, scheduler):

    save_name = os.path.join(args.saved_model, 'model_best.pth')
    print('start tarining...')

    min_loss = 100000000000000000
    train_iterator = iter(train_loader)
    pbar = tqdm.tqdm(total=args.total_steps, ncols=0, desc="Training", unit=" step")
    for step in range(args.total_steps):
        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        try:
            image, label, heatmap_gt = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            image, label, heatmap_gt = next(train_iterator)

        train_loss, gaze_loss, heatmap_loss, landmark_loss = train(args, image, label, heatmap_gt, model, optimizer, step)

        pbar.update()
        pbar.set_postfix(
            heatmap_loss = f"{heatmap_loss.data:.4f}",
            landmarks_loss = f"{landmark_loss.data:.4f}",
            gaze_loss = f"{gaze_loss.data:.4f}",
            total_loss = f"{train_loss.data:.4f}",
            lr=f"{lr:.6f}"
        )

        if (step + 1) % 2000 == 0:
            pbar.close()
            valid_loss = valid(args, model, valid_loader, step)

            if valid_loss <= min_loss:
                min_loss = valid_loss
                torch.save(model.state_dict(), save_name)

            pbar = tqdm.tqdm(total=args.total_steps, ncols=0, desc="total_step", unit=" step")
            pbar.update(step)
        
        scheduler.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ''' Paths '''
    # parser.add_argument('--data_dir', type=str, default="/home/brianw0924/hdd/TEyeD")
    parser.add_argument('--data_dir', type=str, default="../dataset/TEyeD")
    parser.add_argument('--saved_model', type=str, default='./checkpoints/gaze_landmark')
    parser.add_argument('--saved_pred', type=str, default='./test_image')
    parser.add_argument('--load', type=str, default='')
    
    """model parameters"""
    parser.add_argument('--nstack', type=int, default=3)
    parser.add_argument('--nfeatures', type=int, default=32, help='Number of feature maps to use.')
    parser.add_argument('--nlandmarks', type=int, default=8, help='Number of landmarks to be predicted.')

    ''' paramters '''
    parser.add_argument('--image_width', type=int, default=192, help='Image width')
    parser.add_argument('--image_height', type=int, default=144, help='Image height')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--total_steps', type=int, default=100000)
    parser.add_argument('--decay_steps', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    args = parser.parse_args()
    os.makedirs(args.saved_model, exist_ok=True)
    os.makedirs(os.path.join(args.saved_pred, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.saved_pred, 'valid'), exist_ok=True)
    Set_seed(args.seed)

    train_data = Dikablis_data(args, 'train')
    valid_data = Dikablis_data(args, 'valid')

    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False, shuffle=False)
    print('Number of training data : ', len(train_data))
    print('Number of validation data : ', len(valid_data))
    
    model = EyeNet(args, nstack=args.nstack, nfeatures=args.nfeatures, nlandmarks=args.nlandmarks).cuda()
    if args.load:
        model.load_state_dict(torch.load(args.load))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.decay_steps, eta_min=1e-4)

    main(args, model, train_loader, valid_loader, optimizer, scheduler)