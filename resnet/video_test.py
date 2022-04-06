import argparse
import os
from turtle import right
import cv2
from PIL import Image
import glob

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

import tqdm
import time
from utils import *
from unet import UNet
from unet import parameters

import warnings
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
''' Paths '''
parser.add_argument('--load_gaze', type=str, default='./checkpoints/gaze_v2/model_crop.pth')
parser.add_argument('--load_valid', type=str, default='./checkpoints/valid/neurobit_fine_tune.pth')
parser.add_argument('--load_center', type=str, default='./checkpoints/ceneter/neurobit_center.pth')

parser.add_argument('--video_path', type=str, default='../../dataset/test_video')
parser.add_argument('--target_path', type=str, default='./test_video')
parser.add_argument('--output_csv', type=str, default='./output_csv')
parser.add_argument('--threshold', type=float, default=0.5, help='determine whether eyes are open')

''' paramters '''
parser.add_argument('--out_width', type=int, default=320, help='Image width')
parser.add_argument('--out_height', type=int, default=200, help='Image height')
parser.add_argument('--FPS', type=int, default=210, help='Frame per second')
parser.add_argument('--Lefteye_ROI', type=tuple, default=(0, 400, 700, 1130)) # (left, right, top, bot)
parser.add_argument('--Righteye_ROI', type=tuple, default=(0, 400, 150, 580))   # (left, right, top, bot)
parser.add_argument('--seed', type=int, default=17)
args = parser.parse_args()

Set_seed(args.seed)
os.makedirs(args.output_csv, exist_ok=True)
os.makedirs(args.target_path, exist_ok=True)

FPS = args.FPS
IMAGE_WIDTH, IMAGE_HEIGHT = args.Lefteye_ROI[3] - args.Lefteye_ROI[2], args.Lefteye_ROI[1] - args.Lefteye_ROI[0] # frame_size
GAZE_CROP_WIDTH, GAZE_CROP_HEIGHT = 350, 200

# write video parameters
crop_size = (IMAGE_WIDTH, IMAGE_HEIGHT) # (w, h)
merge_size = (crop_size[0]*2, crop_size[1])

# Arrow parameter
color_tan = (0, 255, 0)                 # arrow tangent color
color_sin = (0, 0, 255)                 # arrow sin color
thickness = 4                           # arrow thickness
scaling = 150                           # scaling the arrow length

C_END = "\033[0m"
C_RED = "\033[31m"
C_GREEN = "\033[32m"

class video_test():
    def __init__(self, args, model_gaze, model_valid, model_center):
        self.model_valid = model_valid
        self.model_gaze = model_gaze
        self.model_center = model_center

        self.source = args.video_path
        self.target = args.target_path

        self.Size_X, self.Size_Y = args.out_width, args.out_height # model input size
        self.ratio_x, self.ratio_y = IMAGE_WIDTH / self.Size_X, IMAGE_HEIGHT / self.Size_Y # ratio for center prediction
        self.tfm = transforms.Compose([
            transforms.Resize((self.Size_Y, self.Size_X)), # depends on your model input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.298], std=[0.210]),
        ])

        self.sigmoid = nn.Sigmoid()
        self.threshold = args.threshold

        self.L_top, self.L_bot, self.L_left, self.L_right = args.Lefteye_ROI[0], args.Lefteye_ROI[1], args.Lefteye_ROI[2], args.Lefteye_ROI[3]
        self.R_top, self.R_bot, self.R_left, self.R_right = args.Righteye_ROI[0], args.Righteye_ROI[1], args.Righteye_ROI[2], args.Righteye_ROI[3]

        # Save yaw, pitch
        self.left_yaw = []
        self.left_pitch = []
        self.right_yaw = []
        self.right_pitch = []

        self.Timestamps = []
    
    def center_predction(self, images):
        with torch.no_grad():
            center = self.model_center(images)
            center = center.cpu().detach().numpy()
        center[:, 0], center[:, 1] = center[:, 0] * self.ratio_x, center[:, 1] * self.ratio_y

        return center.astype(np.int)
    
    def valid_prediction(self, images):
        # Valid prediction
        left_valid = True
        right_valid = True

        with torch.no_grad():
            valid = self.model_valid(images)
            valid = self.sigmoid(valid)
            valid = valid.cpu().detach().numpy()
        
        if valid[0][0] < self.threshold:
            left_valid = False

        if valid[1][0] <  self.threshold:
            right_valid = False
        
        return left_valid, right_valid

    def gaze_prediction(self, left_eye, right_eye, center):
        left_center_x, left_center_y = center[0]
        right_center_x, right_center_y = center[1]

        L_top, L_bot, L_left, L_right = left_center_y - 100, left_center_y + 100, left_center_x - 175, left_center_x + 175

        # Boundary
        if L_left < 0:
            L_left = 0
            L_right = GAZE_CROP_WIDTH
        if L_right > IMAGE_WIDTH:
            L_left = IMAGE_WIDTH - GAZE_CROP_WIDTH
            L_right = IMAGE_WIDTH
        if L_top < 0:
            L_top = 0
            L_bot = GAZE_CROP_HEIGHT
        if L_bot > IMAGE_HEIGHT:
            L_bot = IMAGE_HEIGHT
            L_top = IMAGE_HEIGHT - GAZE_CROP_HEIGHT

        left_eye_crop = left_eye[L_top:L_bot, L_left:L_right]
        
        R_top, R_bot, R_left, R_right = right_center_y - 100, right_center_y + 100, right_center_x - 175, right_center_x + 175

        # Boundary
        if R_left < 0:
            R_left = 0
            R_right = GAZE_CROP_WIDTH
        if R_right > IMAGE_WIDTH:
            R_left = IMAGE_WIDTH - GAZE_CROP_WIDTH
            R_right = IMAGE_WIDTH
        if R_top < 0:
            R_top = 0
            R_bot = GAZE_CROP_HEIGHT
        if R_bot > IMAGE_HEIGHT:
            R_bot = IMAGE_HEIGHT
            R_top = IMAGE_HEIGHT - GAZE_CROP_HEIGHT
                
        right_eye_crop = right_eye[R_top:R_bot, R_left:R_right]

        images = torch.cat([
            torch.unsqueeze(self.tfm(Image.fromarray(left_eye_crop).convert('RGB')),dim=0),
            torch.unsqueeze(self.tfm(Image.fromarray(right_eye_crop).convert('RGB')),dim=0)
        ], dim=0)
        
        # Model inference
        with torch.no_grad():
            pred = self.model_gaze(images.cuda()) # [[left_yaw, left_pitch],
                                                  # [right_yaw, right_pitch]]
        return pred
    
    def gaze_visualization(self, eye, pred, center):
        # Project to x-y
        x_tan = - torch.tan(pred[0]*np.pi/180)
        y_tan = - torch.tan(pred[1]*np.pi/180)
        x_sin = - torch.sin(pred[0]*np.pi/180)
        y_sin = - torch.sin(pred[1]*np.pi/180)

        # Draw right arrow
        lt = (int(center[0] + scaling * x_tan.item()), int(center[1] + scaling * y_tan.item())) # arrow ending point
        cv2.arrowedLine(eye,  (center[0], center[1]), lt, (0,255, 0), 8)

        lt = (int(center[0] + scaling * x_sin.item()), int(center[1] + scaling * y_sin.item())) # arrow ending point
        cv2.arrowedLine(eye,  (center[0], center[1]), lt, (0,0,255), 4)

    def main(self):
        for mp4_path in glob.glob(f'{self.source}/*.mp4'):
            save_list = [["Frame", "Yaw(right)", "Pitch(right)", "Yaw(left)", "Pitch(left)"]]

            cap = cv2.VideoCapture(mp4_path)
            print(f'\nInference on {cap.get(cv2.CAP_PROP_FRAME_COUNT):.0f} frames of {mp4_path.split("/")[-1]}')

            # Write video
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            Writer = cv2.VideoWriter(os.path.join(self.target, mp4_path.split('/')[-1]) , fourcc, FPS, merge_size)

            start_time = time.time()
            success = True
            frame_count = 0
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                left_eye = frame[self.L_top:self.L_bot, self.L_left:self.L_right, :]
                right_eye = frame[self.R_top:self.R_bot, self.R_left:self.R_right, :]

                right_eye_flip = right_eye[:, ::-1, :]

                images = torch.cat([
                    torch.unsqueeze(self.tfm(Image.fromarray(left_eye).convert('RGB')),dim=0),
                    torch.unsqueeze(self.tfm(Image.fromarray(right_eye_flip).convert('RGB')),dim=0)
                ], dim=0)
                images = images.cuda()

                center = self.center_predction(images)
                left_valid, right_valid = self.valid_prediction(images)

                gaze_pred = self.gaze_prediction(left_eye, right_eye_flip, center)

                if left_valid:
                    left_yaw, left_pitch = gaze_pred[0].cpu().detach().numpy()
                    self.gaze_visualization(left_eye, gaze_pred[0], center[0])
                else:
                    left_yaw, left_pitch = '', ''
                
                if right_valid:
                    gaze_pred[1][0] = -gaze_pred[1][0]
                    center[1][0] = 430 - center[1][0]
                    right_yaw, right_pitch = gaze_pred[1].cpu().detach().numpy()
                    self.gaze_visualization(right_eye, gaze_pred[1], center[1])
                else:
                    right_yaw, right_pitch = '', ''

                # Merge two eyes to one video
                two_eye = np.concatenate((right_eye, left_eye), axis=1)
                # cv2.imshow("eyes", two_eye)

                # Save info.
                self.left_yaw.append(left_yaw)
                self.left_pitch.append(left_pitch)
                self.right_yaw.append(right_yaw)
                self.right_pitch.append(right_pitch)
                self.Timestamps.append(frame_count)

                Writer.write(two_eye)
                frame_count += 1
                save_list.append([str(frame_count), str(right_yaw), str(right_pitch), str(left_yaw), str(left_pitch)])

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            end_time = time.time()
            print(f'Inference time: {end_time - start_time:.1f} s')
            print(f'Inference speed: {frame_count / (end_time - start_time):.1f} fps')

            save_root = os.path.join(args.output_csv, mp4_path.split('/')[-1][:-4] + '.csv')
            cap.release()
            cv2.destroyAllWindows()
            np.savetxt(save_root,  save_list, fmt='%s', delimiter=',')

if __name__ == '__main__':
    model_valid = models.resnet18(pretrained=True)
    model_valid.fc = nn.Linear(model_valid.fc.in_features, 1)
    if os.path.exists(args.load_valid):
        model_valid.load_state_dict(torch.load(args.load_valid))
        print(C_GREEN + 'Valid check point Successfully Loaded' + C_END)
    else:
        print(C_RED + 'Valid check point Not Found' + C_END)
    model_valid.eval()

    model_gaze = models.resnet18(pretrained=True)
    model_gaze.fc = nn.Linear(model_gaze.fc.in_features, 2)
    if os.path.exists(args.load_gaze):
        model_gaze.load_state_dict(torch.load(args.load_gaze))
        print(C_GREEN + 'Gaze check point Successfully Loaded' + C_END)
    else:
        print(C_RED + 'Gaze check point Not Found' + C_END)
    model_gaze.eval()

    model_center = models.resnet18(pretrained=True)
    model_center.fc = nn.Linear(model_center.fc.in_features, 2)
    if os.path.exists(args.load_center):
        model_center.load_state_dict(torch.load(args.load_center))
        print(C_GREEN + 'Center check point Successfully Loaded' + C_END)
    else:
        print(C_RED + 'Center check point Not Found' + C_END)
    model_center.eval()

    model_gaze = model_gaze.cuda()
    model_valid = model_valid.cuda()
    model_center = model_center.cuda()

    main = video_test(args, model_gaze, model_valid, model_center)
    main.main()


    


