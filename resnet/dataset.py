import os
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random

from utils import vec2angle

class Dikablis_data(data.Dataset):

    def __init__(self, args, mode):
        self.in_w, self.in_h = 384, 288 # Width and height of Dikablis dataset 
        self.out_w, self.out_h = args.image_width, args.image_height

        self.tfm = transforms.Compose([
                    transforms.Resize((self.out_h, self.out_w)), # Since the width and the height of cv2 is (height, width)
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.503], std=[0.224])
                ])
        
        self.data_dir = args.data_dir
        self.dataset = args.dataset
        self.mode = mode

        video_name_file = np.loadtxt(os.path.join(self.data_dir, self.dataset, '%s.txt' % self.mode), delimiter=';', dtype=np.str)
        
        self.image_path = []
        self.gaze_vector_path = []
        for video_name in video_name_file:
            image_path = os.path.join(self.data_dir, self.dataset, video_name, 'frame')

            image_frame_list = sorted(os.listdir(image_path))
            for frame_name in image_frame_list:
                frame_path = os.path.join(image_path, frame_name)
                gaze_path = os.path.join(self.data_dir, self.dataset, video_name, 'gaze_vector', frame_name[:-4] + '.txt')

                self.image_path.append(frame_path)
                self.gaze_vector_path.append(gaze_path)
        assert len(self.image_path) == len(self.gaze_vector_path)

    def __getitem__(self, index):
        image = Image.open(self.image_path[index])
        gaze = np.loadtxt(self.gaze_vector_path[index], delimiter=',', dtype=np.float)
        gaze = vec2angle(gaze) # (x,y,z) to (pitch, yaw)
        
        return self.tfm(image),  torch.FloatTensor(gaze)

    def __len__(self):
        return len(self.image_path)

class Neurobit_data(data.Dataset):
    def __init__(self, args, mode):
        self.out_w, self.out_h = args.image_width, args.image_height
        self.tfm = transforms.Compose([
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
            image_path = os.path.join(self.data_dir, 'image', image_name)

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
