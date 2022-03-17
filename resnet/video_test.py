import argparse
import os
from turtle import right
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

import tqdm
from utils import *


import warnings
warnings.filterwarnings("ignore")

def test_video(args, model_gaze, model_valid):
    tfm = transforms.Compose([
        transforms.Resize((args.image_height, args.image_width)), # depends on your model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.298], std=[0.210]),
    ])

    sigmoid = nn.Sigmoid()
    # Bounding box
    L_top, L_bot, L_left, L_right = args.Lefteye_ROI[0], args.Lefteye_ROI[1], args.Lefteye_ROI[2], args.Lefteye_ROI[3]
    R_top, R_bot, R_left, R_right = args.Righteye_ROI[0], args.Righteye_ROI[1], args.Righteye_ROI[2], args.Righteye_ROI[3]

    save_data = [['yaw(left)', 'pitch(left)', 'yaw(right)', 'pitch(right)']]
    # Iterate thru videos
    video = cv2.VideoCapture(args.video_dir)

    success = True
    count = 0
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
            pred = model_gaze(images.cuda())
            valid = model_valid(images.cuda())
            valid = sigmoid(valid)

            valid = valid.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            left_yaw, left_pitch = str(pred[0][0]), str(pred[0][1])
            right_yaw, right_pitch = str(pred[1][0]), str(pred[1][1])

        if valid[0][0] <args.threshold:
            left_yaw = ''
            left_pitch = ''
            # print('left_eye')
            # cv2.imshow('My Image', np.concatenate((right_eye, left_eye), axis=1))

            # # 按下任意鍵則關閉所有視窗
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        if valid[1][0] < args.threshold:
            right_yaw = ''
            right_pitch = ''
            # print('right_eye')
            # cv2.imshow('My Image', np.concatenate((right_eye, left_eye), axis=1))

            # # 按下任意鍵則關閉所有視窗
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        if count % args.fps == 0:
            print('frame second: ', count // args.fps)
        save_data.append([left_yaw, left_pitch, right_yaw, right_pitch])
    file_name = args.video_dir.split('/')[-1][:-4]
    save_root = os.path.join(args.output_csv, file_name + '.csv')
    np.savetxt(save_root,  save_data, fmt='%s', delimiter=',')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ''' Paths '''
    # parser.add_argument('--data_dir', type=str, default="/home/brianw0924/hdd/TEyeD")
    parser.add_argument('--video_dir', type=str, default="../../20211116_H03_NSS40531/20211116_163103_H03_NSS40531_Test1.mp4")
    parser.add_argument('--load_gaze', type=str, default='./checkpoints/resnet18_SSL_gaze/model_1.9837.pth')
    parser.add_argument('--load_valid', type=str, default='./checkpoints/resnet18_valid/model_best.pth')
    parser.add_argument('--output_csv', type=str, default='./test_csv')  # output csv
    parser.add_argument('--threshold', type=float, default=0.7, help='determine whether eyes are open')

    ''' paramters '''
    parser.add_argument('--image_width', type=int, default=320, help='Image width')
    parser.add_argument('--image_height', type=int, default=200, help='Image height')
    parser.add_argument('--Lefteye_ROI', type=tuple, default=(0, 400, 640, 1280))     # (x, y) <=> (dim=1, dim=0) <=> (w, h)
    parser.add_argument('--Righteye_ROI', type=tuple, default=(0, 400, 0, 640))     # (x, y) <=> (dim=1, dim=0) <=> (w, h)
    # parser.add_argument('--Lefteye_ROI', type=tuple, default=(50, 350, 760, 1160))     # (x, y) <=> (dim=1, dim=0) <=> (w, h)
    # parser.add_argument('--Righteye_ROI', type=tuple, default=(50, 350, 120, 520))     # (x, y) <=> (dim=1, dim=0) <=> (w, h)
    parser.add_argument('--fps', type=int, default=210)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    Set_seed(args.seed)
    os.makedirs(args.output_csv, exist_ok=True)

    model_valid = models.resnet18(pretrained=True)
    model_valid.fc = nn.Linear(model_valid.fc.in_features, 1)
    model_valid.load_state_dict(torch.load(args.load_valid))
    model_valid.eval()

    model_gaze = models.resnet18(pretrained=True)
    model_gaze.fc = nn.Linear(model_gaze.fc.in_features, 2)
    model_gaze.load_state_dict(torch.load(args.load_gaze))
    model_gaze.eval()

    model_gaze = model_gaze.cuda()
    model_valid = model_valid.cuda()
    
    test_video(args, model_gaze, model_valid)


    


