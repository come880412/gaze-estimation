import numpy as np
import argparse
import os
import tqdm
import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import Set_seed
from dataset import Neurobit_data
from model import EyeNet

import warnings
warnings.filterwarnings("ignore")

def valid(args, model, valid_loader, step):
    criterion = nn.MSELoss(reduction='none')
    model.eval()
    pbar = tqdm.tqdm(total=len(valid_loader), ncols=0, desc="Valid[%d/%d]"%(step, args.total_steps), unit=" step")

    total_loss = 0
    yaw_total_loss = 0
    pitch_total_loss = 0
    total_num_image = 0
    with torch.no_grad():
        for idx, (image, gaze) in enumerate(valid_loader):
            image = image.float().cuda()
            gaze = gaze.cuda() # (yaw, pitch)

            _, landmarks_pred, gaze_pred = model(image)

            loss = criterion(gaze_pred, gaze)
            yaw_loss, pitch_loss = loss[:, 0], loss[:, 1]

            total_num_image += len(image)
            yaw_total_loss += torch.sum(yaw_loss)
            pitch_total_loss += torch.sum(pitch_loss)
            total_loss += torch.sum(loss)

            pbar.update()
            pbar.set_postfix(
                yaw_loss = f"{yaw_total_loss / total_num_image:.4f}",
                pitch_loss = f"{pitch_total_loss / total_num_image:.4f}",
                total_loss = f"{total_loss / total_num_image:.4f}"
            )

        image = image.cpu().detach().numpy()[-1]
        image = (image.transpose((1,2,0)) * 0.197) + 0.245

        landmarks_pred = landmarks_pred.cpu().detach().numpy()[-1]
        landmark_image_pred = image.copy()
        for l in landmarks_pred:
            y, x = l
            landmark_image_pred = cv2.circle(landmark_image_pred, (int(round(x * 2)),int(round(y * 2))), 1, (0,0,255), 3)
        cv2.imwrite('./%s/valid/eye.jpg' % args.saved_pred, image * 255)
        cv2.imwrite('./%s/valid/pred_eye_landmark.jpg' % args.saved_pred, landmark_image_pred * 255)

        return total_loss / total_num_image

def train(args, image, gaze, model, optimizer, step):
    criterion = nn.MSELoss(reduction='none')
    model.train()

    image = image.cuda()
    gaze = gaze.cuda() # (yaw, pitch)

    optimizer.zero_grad()
    _, landmarks_pred, gaze_pred = model(image)

    loss = criterion(gaze_pred, gaze)
    yaw_loss, pitch_loss = loss[:, 0], loss[:, 1]

    yaw_loss = torch.mean(yaw_loss)
    pitch_loss = torch.mean(pitch_loss)
    train_loss = torch.mean(loss)

    train_loss.backward()
    optimizer.step()

    if step % 100 == 0:
        image = image.cpu().detach().numpy()[-1]
        image = (image.transpose((1,2,0)) * 0.197) + 0.245

        landmarks_pred = landmarks_pred.cpu().detach().numpy()[-1]
        landmark_image_pred = image.copy()
        for l in landmarks_pred:
            y, x = l
            landmark_image_pred = cv2.circle(landmark_image_pred, (int(round(x * 2)),int(round(y * 2))), 1, (0,0,255), 3)
        cv2.imwrite('./%s/train/eye.jpg' % args.saved_pred, image * 255)
        cv2.imwrite('./%s/train/pred_eye_landmark.jpg' % args.saved_pred, landmark_image_pred * 255)
            
    return train_loss, yaw_loss, pitch_loss

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
            image, gaze = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            image, gaze = next(train_iterator)

        train_loss, yaw_loss, pitch_loss = train(args, image, gaze, model, optimizer, step)

        pbar.update()
        pbar.set_postfix(
            yaw_loss = f"{yaw_loss:.4f}",
            pitch_loss = f"{pitch_loss:.4f}",
            total_loss = f"{train_loss:.4f}"
        )

        if (step + 1) % 5000 == 0:
            pbar.close()
            valid_loss = valid(args, model, valid_loader, step)

            if valid_loss <= min_loss:
                min_loss = valid_loss
                torch.save(model.state_dict(), save_name)

            pbar = tqdm.tqdm(total=args.total_steps, ncols=0, desc="total_step", unit=" step")
            pbar.update(step)
        
        scheduler.step()
    print('minimum_loss: ', min_loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ''' Paths '''
    # parser.add_argument('--data_dir', type=str, default="/home/brianw0924/hdd/TEyeD")
    parser.add_argument('--data_dir', type=str, default="../../dataset/neurobit/dataset_nocrop")
    parser.add_argument('--saved_model', type=str, default='./checkpoints/gaze_fine_tune_video_normalization')
    parser.add_argument('--saved_pred', type=str, default='./test_image')
    parser.add_argument('--load', type=str, default='checkpoints/gaze_landmark_lid_pupil_normalize/model_loss1036.45.pth')
    
    """model parameters"""
    parser.add_argument('--nstack', type=int, default=3)
    parser.add_argument('--nfeatures', type=int, default=32, help='Number of feature maps to use.')
    parser.add_argument('--nlandmarks', type=int, default=25, help='Number of landmarks to be predicted.')

    ''' paramters '''
    parser.add_argument('--image_width', type=int, default=192, help='Image width')
    parser.add_argument('--image_height', type=int, default=144, help='Image height')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--total_steps', type=int, default=100000)
    parser.add_argument('--decay_steps', type=int, default=5000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    args = parser.parse_args()
    os.makedirs(args.saved_model, exist_ok=True)
    os.makedirs(os.path.join(args.saved_pred, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.saved_pred, 'valid'), exist_ok=True)
    Set_seed(args.seed)

    train_data = Neurobit_data(args, 'train')
    valid_data = Neurobit_data(args, 'valid')

    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False, shuffle=False)
    print('Number of training data : ', len(train_data))
    print('Number of validation data : ', len(valid_data))
    
    model = EyeNet(args, nstack=args.nstack, nfeatures=args.nfeatures, nlandmarks=args.nlandmarks).cuda()

    if args.load:
        model.load_state_dict(torch.load(args.load))

    # ct = 0
    # for child in model.children():
    #     ct += 1
    #     if ct < 8:
    #         for param in child.parameters():
    #             param.requires_grad = False

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.decay_steps, eta_min=1e-4)

    main(args, model, train_loader, valid_loader, optimizer, scheduler)