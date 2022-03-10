import numpy as np
import argparse
import os
import tqdm
from PIL import Image
import cv2
import math
import json
import torch.nn as nn

import torch
from torch.utils.data import DataLoader
from dataset import Neurobit_data
import torchvision.transforms as transforms

from utils import Set_seed
from model import EyeNet

import warnings
warnings.filterwarnings("ignore")

def get_yaw_pitch(i, h, w, c, d):
    top_left_y = 4 * h + c[1]
    top_left_x = -6 * w + c[0]
    pitch = math.atan( (top_left_y - (i//13) * h) / d) * 180 / math.pi
    yaw   = math.atan( (top_left_x + (i%13)  * w) / d) * 180 / math.pi
    gaze = [yaw, pitch]
    gaze = torch.stack((torch.FloatTensor(gaze), torch.FloatTensor(gaze)), dim=0)
    return gaze

def cal_loss(pred, true):
    criterion = nn.MSELoss(reduction='none')
    return criterion(pred, true)

def Inference_and_visualization(args, model):
    tfm = transforms.Compose([
        transforms.Resize((args.image_height, args.image_width)), # depends on your model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.245], std=[0.197])
    ])

    # Bounding box
    L_top, L_bot, L_left, L_right = args.Lefteye_ROI[0], args.Lefteye_ROI[1], args.Lefteye_ROI[2], args.Lefteye_ROI[3]
    R_top, R_bot, R_left, R_right = args.Righteye_ROI[0], args.Righteye_ROI[1], args.Righteye_ROI[2], args.Righteye_ROI[3]

    # Video size
    crop_size = (L_right-L_left, L_bot-L_top) # (w, h)
    merge_size = (crop_size[0], crop_size[1])

    # Arrow parameter
    color = (0, 255, 0)                     # arrow color
    thickness = 4                           # arrow thickness
    scaling = 150                           # scaling the arrow length
    s = (crop_size[0]//2, crop_size[1]//2)  # arrow starting point

    with open(os.path.join(args.data_info_dir)) as f:
        data = json.load(f)
    args.d = data['distance_to_grid']

    eye = args.eye # righteye or lefteye or both

    count = 0
    left_yaw_total = 0
    left_pitch_total = 0
    right_yaw_total = 0
    right_pitch_total = 0
    for idx, file in tqdm.tqdm(enumerate(sorted(os.listdir(args.data_dir)))):
        video = cv2.VideoCapture(os.path.join(args.data_dir, file))

        gaze = get_yaw_pitch(i=float(idx),h=args.h, w=args.w, c=args.c, d=args.d) # length-based label
        if eye == 'both':
            gaze_list = [['yaw(left),pitch(left),yaw(right),pitch(right)']]
        else:
            gaze_list = [['yaw, pitch']]

        # Video parameter
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        # LeftEye_Writer  = cv2.VideoWriter(os.path.join(args.output_video,f"{file.split('.')[0]}_left.mp4") , fourcc, args.fps, crop_size)
        # RightEye_Writer = cv2.VideoWriter(os.path.join(args.output_video,f"{file.split('.')[0]}_right.mp4") , fourcc, args.fps, crop_size)
        Writer = cv2.VideoWriter(os.path.join(args.output_video, file) , fourcc, args.fps, merge_size)

        # Inference
        success = True
        while success:
            success, frame = video.read()
            if not success:
                break
            count+=1
            left_eye, right_eye = frame[L_top:L_bot, L_left:L_right, :], frame[R_top:R_bot, R_left:R_right, :]

            # Model input
            images = torch.cat([
                    torch.unsqueeze(tfm(Image.fromarray(left_eye).convert('RGB')),dim=0),
                    torch.unsqueeze(tfm(Image.fromarray(right_eye).convert('RGB')),dim=0)
                ], dim=0
            )

            # Model inference
            with torch.no_grad():
                _, _, gaze_pred = model(images.cuda())
                gaze = gaze.cuda()
                gaze_loss = cal_loss(gaze_pred, gaze.cuda())
                left_yaw_loss, left_pitch_loss = gaze_loss[0, 0], gaze_loss[0, 1]
                right_yaw_loss, right_pitch_loss = gaze_loss[1, 0], gaze_loss[1, 1]

                right_yaw_total += right_yaw_loss.data
                right_pitch_total += right_pitch_loss.data
                left_yaw_total += left_yaw_loss.data
                left_pitch_total += left_pitch_loss.data
            
            gaze_pred = gaze_pred.cpu().detach()
            # Project to x-y
            if eye == 'left':
                gaze_list.append([f'{str(gaze_pred[0,0].item())}, {str(gaze_pred[0,1].item())}'])
                lx = - torch.sin(gaze_pred[0,0]*np.pi/180)
                ly = - torch.sin(gaze_pred[0,1]*np.pi/180)
                lt = (int(s[0] + scaling * lx.item()), int(s[1] + scaling * ly.item())) # arrow ending point
                # Draw arrow
                eye_arrow  = cv2.arrowedLine(left_eye,  s, lt, color, thickness)

            elif eye == 'right':
                gaze_list.append([f'{str(gaze_pred[1,0].item())}, {str(gaze_pred[1,1].item())}'])
                rx = - torch.sin(gaze_pred[1,0]*np.pi/180)
                ry = - torch.sin(gaze_pred[1,1]*np.pi/180)
                rt = (int(s[0] + scaling * rx.item()), int(s[1] + scaling * ry.item())) # arrow ending point
                # Draw arrow
                eye_arrow = cv2.arrowedLine(right_eye, s, rt, color, thickness)
            
            elif eye == 'both':
                gaze_list.append([f'{str(gaze_pred[0,0].item())}, {str(gaze_pred[0,1].item())}, \
                                 {str(gaze_pred[1,0].item())}, {str(gaze_pred[1,1].item())}'])
                lx = - torch.sin(gaze_pred[0,0]*np.pi/180)
                ly = - torch.sin(gaze_pred[0,1]*np.pi/180)
                rx = - torch.sin(gaze_pred[1,0]*np.pi/180)
                ry = - torch.sin(gaze_pred[1,1]*np.pi/180)
                lt = (int(s[0] + scaling * lx.item()), int(s[1] + scaling * ly.item())) # arrow ending point
                rt = (int(s[0] + scaling * rx.item()), int(s[1] + scaling * ry.item())) # arrow ending point
                # Draw arrow
                left_eye  = cv2.arrowedLine(left_eye,  s, lt, color, thickness)
                right_eye = cv2.arrowedLine(right_eye, s, rt, color, thickness)
                # Merge two eyes to one video
                eye_arrow = np.concatenate((right_eye, left_eye), axis=1)
            
            Writer.write(eye_arrow)
        video.release()
        Writer.release()
        cv2.destroyAllWindows()
        np.savetxt(os.path.join(args.output_csv, '%s.csv' % file[:-4]),  gaze_list, fmt='%s', delimiter=',')
    print('%s | left_yaw: %.4f | left_pitch: %.4f | right_yaw: %.4f | right_pitch: %.4f' 
                %(file, left_yaw_total / count, left_pitch_total /count, right_yaw_total / count, right_pitch_total / count))

def test(args, model, loader, mode):
    pbar = tqdm.tqdm(total=len(loader), ncols=0, desc="%s" % mode, unit=" step")

    total_loss = 0
    yaw_total_loss = 0
    pitch_total_loss = 0
    total_num_image = 0
    with torch.no_grad():
        for idx, (image, gaze) in enumerate(loader):
            image = image.float().cuda()
            gaze = gaze.cuda() # (yaw, pitch)
            total_num_image += len(image)

            _, landmarks_pred, gaze_pred = model(image)

            loss = cal_loss(gaze_pred, gaze)
            yaw_loss, pitch_loss = loss[:, 0], loss[:, 1]

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
        cv2.imwrite('./%s/eye.jpg' % args.save_pred, image * 255)
        cv2.imwrite('./%s/pred_eye_landmark.jpg' % args.save_pred, landmark_image_pred * 255)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ''' Paths '''
    # parser.add_argument('--data_dir', type=str, default="/home/brianw0924/hdd/TEyeD")
    parser.add_argument('--data_dir', type=str, default="../../dataset/neurobit")
    parser.add_argument('--output_video', type=str, default='./test_result')    # output video
    parser.add_argument('--output_csv', type=str, default='./test_csv')         # output csv
    parser.add_argument('--save_pred', type=str, default='./test_image') 
    parser.add_argument('--load', type=str, default='./checkpoints/gaze_8folders/model_2.01.pth')
    parser.add_argument('--mode', type=str, default='test', help='valid/test')

    """ data_info """
    parser.add_argument('--eye', type=str, default='left', help='right/eye/both')
    parser.add_argument('--data_info_dir', type=str, default="../Neurobit_data/20220121_H14_NSS00121.json")
    parser.add_argument('--h', type=float, default=6.8)   
    parser.add_argument('--w', type=float, default=7.7)
    parser.add_argument('--c', type=tuple, default=(0,0)) # (x,y)
    parser.add_argument('--d', type=float, default=78)    # offset

    """model parameters"""
    parser.add_argument('--nstack', type=int, default=3)
    parser.add_argument('--nfeatures', type=int, default=32, help='Number of feature maps to use.')
    parser.add_argument('--nlandmarks', type=int, default=25, help='Number of landmarks to be predicted.')

    ''' paramters '''
    parser.add_argument('--image_width', type=int, default=192, help='Image width')
    parser.add_argument('--image_height', type=int, default=144, help='Image height')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--Lefteye_ROI', type=tuple, default=(50, 350, 760, 1160))     # (x, y) <=> (dim=1, dim=0) <=> (w, h)
    parser.add_argument('--Righteye_ROI', type=tuple, default=(50, 350, 120, 520))     # (x, y) <=> (dim=1, dim=0) <=> (w, h)
    parser.add_argument('--fps', type=int, default=210)

    args = parser.parse_args()
    Set_seed(args.seed)
    os.makedirs(args.output_video, exist_ok=True)
    os.makedirs(args.output_csv, exist_ok=True)
    os.makedirs(args.save_pred, exist_ok=True)
    
    dataset = Neurobit_data(args, args.mode)
    loader = DataLoader(dataset, batch_size=128, num_workers=0, drop_last=False, shuffle=False)

    model = EyeNet(args, nstack=args.nstack, nfeatures=args.nfeatures, nlandmarks=args.nlandmarks).cuda()
    if args.load:
        print('Load model!!')
        model.load_state_dict(torch.load(args.load))
    model.eval()
    model = model.cuda()
    
    # Inference_and_visualization(args, model)
    test(args, model, loader, args.mode)
    


