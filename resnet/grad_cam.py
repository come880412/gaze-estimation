import os
import torchvision.models as models
from pytorch_grad_cam import GradCAM
import tqdm
from dataset import Neurobit_data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import argparse
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image

def grad_cam(input_tensor, model):
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    target_category = None

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category, aug_smooth=True)

    return grayscale_cam

def visualize_layer(opt, model, test_loader):
    sigmoid = nn.Sigmoid()

    model.eval()
    for image, label, file_path in tqdm.tqdm(test_loader):
        image = image.cuda()
        pred = model(image)
        if opt.task == 'valid':
            pred = sigmoid(pred)
        pred_label = pred.cpu().detach().numpy()
        label = label.numpy()

        grayscale_cam = grad_cam(image, model)
        for i, gray_cam in enumerate(grayscale_cam):
            print('pred result: ', pred_label[i], '\t label: ', label[i])

            file_path_ = file_path[i]
            image_bgr = cv2.imread(file_path_)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            gray_cam = cv2.resize(gray_cam, (640, 400))
            image_ori = image_rgb.copy()

            image_rgb = (image_rgb / 255.0).astype(np.float32)
            visualization = show_cam_on_image(image_rgb, gray_cam, use_rgb=True)
            plt.subplot(121)
            plt.imshow(visualization)
            plt.subplot(122)
            plt.imshow(image_ori)
            plt.show()

def video_grad_cam(opt, model):
    tfm = transforms.Compose([
        transforms.Resize((opt.image_height, opt.image_width)), # depends on your model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.298], std=[0.210]),
    ])

    sigmoid = nn.Sigmoid()
    # Bounding box
    L_top, L_bot, L_left, L_right = opt.Lefteye_ROI[0], opt.Lefteye_ROI[1], opt.Lefteye_ROI[2], opt.Lefteye_ROI[3]
    R_top, R_bot, R_left, R_right = opt.Righteye_ROI[0], opt.Righteye_ROI[1], opt.Righteye_ROI[2], opt.Righteye_ROI[3]

    video = cv2.VideoCapture(opt.video_dir)
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

        with torch.no_grad():
            pred = model(images.cuda())
            if opt.task == 'valid':
                pred = sigmoid(pred)
            pred = pred.cpu().detach().numpy()
        
        # if pred[1][0] < opt.threshold or pred[0][0] < opt.threshold:
        print(pred)
        grayscale_cam = grad_cam(images.cuda(), model)
        left_eye_cam, right_eye_cam = grayscale_cam[0], grayscale_cam[1]
        left_eye_rgb = cv2.cvtColor(left_eye, cv2.COLOR_BGR2RGB)
        right_eye_rgb = cv2.cvtColor(right_eye, cv2.COLOR_BGR2RGB)
        left_eye_rgb = cv2.resize(left_eye_rgb, (640, 400))
        right_eye_rgb = cv2.resize(right_eye_rgb, (640, 400))

        left_eye_cam = cv2.resize(left_eye_cam, (640, 400))
        right_eye_cam = cv2.resize(right_eye_cam, (640, 400))

        left_eye_rgb = (left_eye_rgb / 255.0).astype(np.float32)
        right_eye_rgb = (right_eye_rgb / 255.0).astype(np.float32)

        left_visualization = show_cam_on_image(left_eye_rgb, left_eye_cam, use_rgb=True)
        right_visualization = show_cam_on_image(right_eye_rgb, right_eye_cam, use_rgb=True)
        
        visualization = np.concatenate((right_visualization, left_visualization), axis=1)

        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.imshow(visualization)
        plt.subplot(122)
        plt.imshow(frame)
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="../../dataset/neurobit")
    parser.add_argument('--video_dir', type=str, default="../../20211116_H03_NSS40531/20211116_163103_H03_NSS40531_Test1.mp4")

    parser.add_argument('--load', type=str, default='./checkpoints/resnet18_SSL_gaze/model_1.9837.pth')
    parser.add_argument('--task', type=str, default='gaze', help='valid/gaze')
    parser.add_argument('--mode', type=str, default='test', help='valid/test')
    parser.add_argument('--threshold', type=float, default=0.5, help='determine whether eyes are open')

    ''' paramters '''
    parser.add_argument('--image_width', type=int, default=320, help='Image width')
    parser.add_argument('--image_height', type=int, default=200, help='Image height')
    parser.add_argument('--Lefteye_ROI', type=tuple, default=(0, 400, 640, 1280))     # (x, y) <=> (dim=1, dim=0) <=> (w, h)
    parser.add_argument('--Righteye_ROI', type=tuple, default=(0, 400, 0, 640))     # (x, y) <=> (dim=1, dim=0) <=> (w, h)
    # parser.add_argument('--Lefteye_ROI', type=tuple, default=(50, 338, 760, 1144))
    # parser.add_argument('--Righteye_ROI', type=tuple, default=(50, 338, 120, 504))
    opt = parser.parse_args()

    random_seed_general = 500
    random.seed(random_seed_general) 
    torch.manual_seed(random_seed_general)
    torch.cuda.manual_seed_all(random_seed_general)
    np.random.seed(random_seed_general)
    random.seed(random_seed_general)
    torch.backends.cudnn.deterministic = True
    
    model = models.resnet18(pretrained=True)
    if opt.task == 'gaze':
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif opt.task == 'valid':
        model.fc = nn.Linear(model.fc.in_features, 1)
    
    model.load_state_dict(torch.load(opt.load))
    model.eval()
    model = model.cuda()
    

    # test_data = Neurobit_data(opt, opt.mode)
    # test_loader = DataLoader(test_data, batch_size=4, num_workers=0, drop_last=False, shuffle=True)
    # visualize_layer(opt, model, test_loader)

    video_grad_cam(opt, model)