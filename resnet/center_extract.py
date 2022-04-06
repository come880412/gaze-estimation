from dataclasses import replace
from random import random
from turtle import color
import numpy as np
import argparse
import os
import cv2
from PIL import Image
import glob
import time
import tqdm

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data

from unet import UNet
from unet import parameters

Size_X = parameters.Size_X
Size_Y = parameters.Size_Y
FPS = parameters.FPS

def Unet_test(image, model, Size_X, Size_Y):
    batch_size = 1
    t = []
    t1 = time.time()
    inputImg = []
    inputImg_BK = []
    center_list = []
    
        
    image = cv2.resize(image, (Size_X, Size_Y), interpolation=cv2.INTER_CUBIC)
    inputImg_BK.append(image.copy())  
    image = image.astype(np.float32)/255
    image = image[np.newaxis, :]

    inputImg.append(image)
    
    inputImg = np.array(inputImg)
        
    t2 = time.time()
    t.append(t2-t1)
    
    tf_images = torch.from_numpy(inputImg)
    tf_images = tf_images.cuda()
    output = model(tf_images)
    output_bk = output[:, 0].clone().detach().cpu().numpy()
    
    t3 = time.time()
    t.append(t3-t2)
    
    output_index = []
    for i in range(batch_size):
        temp = output_bk[i]
        gt_temp = inputImg_BK[i]
        ttt = output_bk[i]
        ttt[ttt < 0.5] = 0
        ttt[ttt >= 0.5] = 1
        if np.count_nonzero(ttt) == 0:
            temp[temp < 0.25] = 0
            temp[temp >= 0.25] = 1
        else:
            temp[temp < 0.5] = 0
            temp[temp >= 0.5] = 1
    
        ## Connected Component Analysis
        if np.count_nonzero(temp) != 0:
            _, labels, stats, center = cv2.connectedComponentsWithStats(temp[:, :].astype(np.uint8))
    
            stats = stats[1:, :]
            pupil_candidate = np.argmax(stats[:, 4]) + 1
            temp[:, :][labels != pupil_candidate] = 0
            
        gt_temp[temp == 1] = 255
        output_bk[i] = temp

        indices = np.argwhere(temp == 1)
        output_index.append(indices)
        if len(indices):
            x_center = np.average(indices[:,0])
            y_center = np.average(indices[:,1])
            center_list.append([x_center, y_center])
        else:
            center_list.append([0, 0])

    t4 = time.time()
    t.append(t4-t3)
    
    return inputImg_BK, t, center_list, output_index

if __name__ == '__main__':
    root = '../../dataset/neurobit/image'
    save_path = '../../dataset/neurobit'

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'center_vis'), exist_ok=True)

    image_width, image_height = 430, 400
    ratio_x, ratio_y = image_width / Size_X, image_height / Size_Y 
    name_load_model = 'trained_model/UNet/'
    cross_val_num = 18
    model_center = UNet(n_channels=1, n_classes=1, bilinear=True)
    if os.path.exists(name_load_model):
        load_saved_model_name = parameters.find_latest_model_name(name_load_model, cross_val_num)
        model_center.load_state_dict(torch.load(load_saved_model_name))
        print(parameters.C_GREEN + 'Check point Successfully Loaded' + parameters.C_END)
    model_center.eval()
    model_center = model_center.cuda()

    data_name_list = os.listdir(root)
    data_name_list.sort()

    center_save_list = []
    for idx, frame_name in enumerate(tqdm.tqdm(data_name_list)):
        frame_path = os.path.join(root, frame_name)
        frame = cv2.imread(frame_path, 1)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        images_center, times, center_list, output_index = Unet_test(frame_gray, model_center, Size_X, Size_Y)

        center_y, center_x = ratio_y * center_list[0][0], ratio_x * center_list[0][1] # (center_y, center_x)
        cv2.circle(frame, (int(center_x), int(center_y)), 3, (0,0,255), thickness=-1)

        if center_x == 0 and center_y == 0:
            print('drop out: ', frame_name)
            continue
        else:
            if idx % 180 == 0:
                cv2.imwrite(os.path.join(save_path, 'center_vis', f'{str(idx).zfill(7)}.png'), frame)
            center_save_list.append([frame_name, center_x, center_y])
    
    center_save_list = np.array(center_save_list)
    random_index = np.random.choice(len(center_save_list), len(center_save_list), replace=False)
    
    train_data = center_save_list[:int(len(random_index) * 0.9)]
    valid_data = center_save_list[int(len(random_index) * 0.9):]

    np.savetxt(os.path.join(save_path, 'train_center.txt'),  train_data, fmt='%s', delimiter=',')
    np.savetxt(os.path.join(save_path, 'valid_center.txt'),  valid_data, fmt='%s', delimiter=',')

        
    