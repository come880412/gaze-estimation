import os
import numpy as np
import random
from PIL import Image
from random import shuffle
import cv2

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from utils import get_heatmaps, vec2angle

class Dikablis_data(data.Dataset):

    def __init__(self, args, mode):
        self.in_w, self.in_h = 384, 288
        self.out_w, self.out_h = args.image_width, args.image_height
    
        self.heatmap_w = int(self.out_w / 2)
        self.heatmap_h = int(self.out_h / 2)

        self.scale_w, self.scale_h = self.heatmap_w / self.in_w, self.heatmap_h / self.in_h

        self.tfm = transforms.Compose([
                    transforms.Resize((self.out_h, self.out_w)), # Since the width and the height of cv2 is (height, width)
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.503], std=[0.224])
                ])
        
        self.data_dir = args.data_dir
        self.mode = mode

        video_name_file = np.loadtxt(os.path.join(self.data_dir, '%s.txt' % self.mode), delimiter=';', dtype=np.str)
        
        self.image_path = []
        self.lid_landmark_path = []
        self.pupil_landmark_path = []
        self.gaze_vector_path = []
        for video_name in video_name_file:
            image_path = os.path.join(self.data_dir, video_name, 'frame')

            image_frame_list = sorted(os.listdir(image_path))
            for idx, frame_name in enumerate(image_frame_list):
                frame_path = os.path.join(image_path, frame_name)
                gaze_path = os.path.join(self.data_dir, video_name, 'gaze_vector', frame_name[:-4] + '.txt')
                lid_landmark_path = os.path.join(self.data_dir, video_name, 'lid_landmark', frame_name[:-4] + '.txt')
                pupil_landmark_path = os.path.join(self.data_dir, video_name, 'pupil_landmark', frame_name[:-4] + '.txt')

                self.image_path.append(frame_path)
                self.lid_landmark_path.append(lid_landmark_path)
                self.pupil_landmark_path.append(pupil_landmark_path)
                self.gaze_vector_path.append(gaze_path)
        assert len(self.image_path) == len(self.pupil_landmark_path) == len(self.gaze_vector_path)

    def __getitem__(self, index):
        image = Image.open(self.image_path[index])
        label_info, heatmap = self.label_processing(self.gaze_vector_path[index], self.lid_landmark_path[index], self.pupil_landmark_path[index])
        
        return self.tfm(image),  torch.FloatTensor(label_info), torch.FloatTensor(heatmap)

    def __len__(self):
        return len(self.image_path)
    
    def label_processing(self, gaze_path, lid_path, pupil_path):
        gaze = np.loadtxt(gaze_path, delimiter=',', dtype=np.float)
        lid_landmark = np.loadtxt(lid_path, delimiter=',', dtype=np.float)
        pupil_landmark = np.loadtxt(pupil_path, delimiter=',', dtype=np.float)
        
        gaze = vec2angle(gaze) # (x,y,z) to (yaw, pitch)

        lid_landmarks = []
        for i in range(0, len(lid_landmark), 4): # reduce lid_landmark points
            lid_landmarks.append(lid_landmark[i])
            lid_landmarks.append(lid_landmark[i+1])
        lid_landmark = lid_landmarks
        
        landmark = np.concatenate((lid_landmark, pupil_landmark))
        landmarks = []
        # Swap columns so that landmarks are in (y, x), not (x, y)
        # This is because the network outputs landmarks as (y, x) values.
        for i in range(0, len(landmark), 2):
            x, y = landmark[i] * self.scale_w, landmark[i+1] * self.scale_h
            landmarks.append([y, x])

        heatmaps = get_heatmaps(w=self.heatmap_w, h=self.heatmap_h, landmarks=landmarks) # (8, heatmap_h, heatmap_w)

        label_info = np.concatenate((gaze, landmarks)).tolist() # 0:gaze, 1~8:pupil
        return label_info, heatmaps
        
class Neurobit_data(data.Dataset):
    def __init__(self, args, mode):
        self.out_w, self.out_h = args.image_width, args.image_height
        self.tfm = transforms.Compose([
                    transforms.Resize((self.out_h, self.out_w)), # Since the width and the height of cv2 is (height, width)
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.298], std=[0.210])
                ])
        self.data_dir = args.data_dir
        self.mode = mode

        data_info = np.loadtxt(os.path.join(self.data_dir, '%s.txt' % self.mode), delimiter=';', dtype=np.str)[1:]

        self.image_path = []
        self.gaze_vector = []
        for info in data_info:
            image_name, yaw, pitch = info.split(',')
            if mode == 'train' or 'valid':
                image_path = os.path.join(self.data_dir, 'image', image_name)
            elif mode == 'test':
                image_path = os.path.join(self.data_dir, 'image_test', image_name)

            self.image_path.append(image_path)
            self.gaze_vector.append([float(yaw), float(pitch)])
        assert len(self.image_path) == len(self.gaze_vector)
    
    def __getitem__(self, index):
        image = Image.open(self.image_path[index])

        if self.mode == 'valid' or self.mode == 'test':
            image = image.crop((120, 50, 520, 350)) # (left, top, right ,bot)
        else:
            vertical = random.randint(-30,30)
            horizontal = random.randint(-40,40)
            image = image.crop((120+horizontal, 50+vertical, 520+horizontal, 350+vertical))
        
        return self.tfm(image),  torch.FloatTensor(self.gaze_vector[index])
    
    def __len__(self):
        return len(self.image_path)

def Dikablis_Norm(data_path='../dataset/TEyeD'): # mean=[0.503], std=[0.224]
    # img_h, img_w = 32, 32
    img_h, img_w = 288, 384   #根据自己数据集适当调整，影响不大
    means, stdevs = [], []
    img_list = []
    
    folder_name = os.listdir(data_path)
    cancel_list = ['train.txt', 'valid.txt']
    imgs_path_list = []
    for folder in folder_name:
        if folder not in cancel_list:
            frame_path = os.path.join(data_path, folder, 'frame')
            frame_list = os.listdir(frame_path)
            shuffle(frame_list)
            frame_list = frame_list[:15]

            for frame in frame_list:
                img_path = os.path.join(frame_path, frame)
                imgs_path_list.append(img_path)
    
    len_ = len(imgs_path_list)
    i = 0
    for path in imgs_path_list:
        img = cv2.imread(path)
        img = cv2.resize(img, (img_w, img_h))
        img = img[:, :, :, np.newaxis]
        img_list.append(img)
        i += 1
        if i%1000 == 0:
            print(i,'/',len_)    
    
    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255.
    
    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))
    
    # BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
    means.reverse()
    stdevs.reverse()
    
    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))

def Neurobit_Norm(data_path='../../dataset/neurobit/image'): # mean=[0.298], std=[0.210]
    # img_h, img_w = 32, 32
    img_h, img_w = 400, 640   #根据自己数据集适当调整，影响不大
    means, stdevs = [], []
    img_list = []
    
    image_name_list = os.listdir(data_path)
    shuffle(image_name_list)
    image_name_list = image_name_list[:5000]
    imgs_path_list = []
    for image_name in image_name_list:
        img_path = os.path.join(data_path, image_name)
        imgs_path_list.append(img_path)
    
    len_ = len(imgs_path_list)
    i = 0
    for path in imgs_path_list:
        img = cv2.imread(path)          
        img = cv2.resize(img, (img_w, img_h))
        img = img[:, :, :, np.newaxis]
        img_list.append(img)
        i += 1
        if i%1000 == 0:
            print(i,'/',len_)    
    
    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255.
    
    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))
    
    # BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
    means.reverse()
    stdevs.reverse()
    
    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))

if __name__ == '__main__':
    # Dikablis_Norm()
    Neurobit_Norm()
    # print(torch.normal(0.0, 1.0, size=(1, 3)))