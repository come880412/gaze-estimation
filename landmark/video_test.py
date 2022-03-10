import argparse
import os
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

import tqdm
from utils import *
from model import EyeNet


import warnings
warnings.filterwarnings("ignore")

def test_video(args, model):
    tfm = transforms.Compose([
        transforms.Resize((args.image_height, args.image_width)), # depends on your model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.298], std=[0.210]),
    ])

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
            _, _, pred = model(images.cuda())
            pred = pred.cpu().detach().numpy()
            left_yaw, left_pitch = str(pred[0][0]), str(pred[0][1])
            right_yaw, right_pitch = str(pred[1][0]), str(pred[1][1])
        
        save_data.append([left_yaw, left_pitch, right_yaw, right_pitch])
    file_name = args.video_dir.split('/')[-1][:-4]
    save_root = os.path.join(args.output_csv, file_name + '.csv')
    np.savetxt(save_root,  save_data, fmt='%s', delimiter=',')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ''' Paths '''
    # parser.add_argument('--data_dir', type=str, default="/home/brianw0924/hdd/TEyeD")
    parser.add_argument('--video_dir', type=str, default="../../20211116_H03_NSS40531/20211116_163103_H03_NSS40531_Test1.mp4")
    parser.add_argument('--load', type=str, default='./checkpoints/gaze_8folders/model_2.01.pth')
    parser.add_argument('--output_csv', type=str, default='./test_csv')  # output csv

    """model parameters"""
    parser.add_argument('--nstack', type=int, default=3)
    parser.add_argument('--nfeatures', type=int, default=32, help='Number of feature maps to use.')
    parser.add_argument('--nlandmarks', type=int, default=25, help='Number of landmarks to be predicted.')

    ''' paramters '''
    parser.add_argument('--image_width', type=int, default=192, help='Image width')
    parser.add_argument('--image_height', type=int, default=144, help='Image height')
    parser.add_argument('--Lefteye_ROI', type=tuple, default=(50, 350, 760, 1160))     # (x, y) <=> (dim=1, dim=0) <=> (w, h)
    parser.add_argument('--Righteye_ROI', type=tuple, default=(50, 350, 120, 520))     # (x, y) <=> (dim=1, dim=0) <=> (w, h)
    parser.add_argument('--fps', type=int, default=210)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    Set_seed(args.seed)
    os.makedirs(args.output_csv, exist_ok=True)

    model = EyeNet(args, nstack=args.nstack, nfeatures=args.nfeatures, nlandmarks=args.nlandmarks).cuda()
    if args.load:
        print('Load model!!')
        model.load_state_dict(torch.load(args.load))
    model.eval()
    model = model.cuda()
    
    test_video(args, model)


    


