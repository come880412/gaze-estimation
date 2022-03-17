import os
import numpy as np
from PIL import Image, ImageOps

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class Dikablis_data(data.Dataset):

    def __init__(self, args, mode):
        self.in_w, self.in_h = 384, 288 # Width and height of Dikablis dataset 
        self.out_w, self.out_h = args.image_width, args.image_height

        self.tfm = transforms.Compose([
            transforms.Resize((self.out_h, self.out_w)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.3, 0.3, 0.3, 0.0)],
                p = 0.5
            ),
            # transforms.RandomGrayscale(p=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.GaussianBlur((5, 5), (1.5, 1.5))],
                p = 0.4
            ),
            # Solarization(0.4),
            # transforms.RandomResizedCrop((200, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.298], std=[0.210]),
        ])
        
        self.data_dir = args.data_dir
        self.mode = mode

        video_name_file = np.loadtxt(os.path.join(self.data_dir, 'TEyeD', '%s.txt' % self.mode), delimiter=';', dtype=np.str)
        
        self.image_path = []
        for video_name in video_name_file:
            image_path = os.path.join(self.data_dir, video_name, 'frame')

            image_frame_list = sorted(os.listdir(image_path))
            for frame_name in image_frame_list:
                frame_path = os.path.join(image_path, frame_name)
                self.image_path.append(frame_path)
        
        valid_data = os.listdir(os.path.join(self.data_dir, 'TEyeD_valid', 'frame'))
        for frame_name in valid_data:
            frame_path = os.path.join(self.data_dir, 'TEyeD_valid', 'frame', frame_name)
            self.image_path.append(frame_path)

    def __getitem__(self, index):
        image = Image.open(self.image_path[index]).convert('RGB')
        
        return self.tfm(image), self.tfm(image)
    def __len__(self):
        return len(self.image_path)


class Dikablis_Neurobit_data(data.Dataset):

    def __init__(self, args, mode):
        self.in_w, self.in_h = 384, 288 # Width and height of Dikablis dataset 
        self.out_w, self.out_h = args.image_width, args.image_height

        self.tfm = transforms.Compose([
            transforms.Resize((self.out_h, self.out_w)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.3, 0.3, 0.3, 0.0)],
                p = 0.5
            ),
            # transforms.RandomGrayscale(p=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.GaussianBlur((5, 5), (1.5, 1.5))],
                p = 0.4
            ),
            # Solarization(0.4),
            # transforms.RandomResizedCrop((200, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.298], std=[0.210]),
        ])
        
        self.data_dir = args.data_dir
        self.mode = mode

        video_name_file = np.loadtxt(os.path.join(self.data_dir, 'TEyeD', '%s.txt' % self.mode), delimiter=';', dtype=np.str)
        
        self.image_path = []
        for video_name in video_name_file:
            image_path = os.path.join(self.data_dir, 'TEyeD', video_name, 'frame')

            image_frame_list = sorted(os.listdir(image_path))
            for frame_name in image_frame_list:
                frame_path = os.path.join(image_path, frame_name)
                self.image_path.append(frame_path)
        
        valid_data = os.listdir(os.path.join(self.data_dir, 'TEyeD_valid', 'frame'))
        for frame_name in valid_data:
            frame_path = os.path.join(self.data_dir, 'TEyeD_valid', 'frame', frame_name)
            self.image_path.append(frame_path)

        data_info = np.loadtxt(os.path.join(self.data_dir, 'neurobit', 'train.txt'), delimiter=';', dtype=np.str)[1:]
        for info in data_info:
            image_name, _, _ = info.split(',')
            image_path = os.path.join(self.data_dir, 'neurobit', 'image', image_name)

            self.image_path.append(image_path)

    def __getitem__(self, index):
        image = Image.open(self.image_path[index]).convert('RGB')
        
        return self.tfm(image), self.tfm(image)
    def __len__(self):
        return len(self.image_path)

class Neurobit_data(data.Dataset):

    def __init__(self, args):
        self.out_w, self.out_h = args.image_width, args.image_height

        self.tfm = transforms.Compose([
            transforms.Resize((self.out_h, self.out_w)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.3, 0.3, 0.3, 0.0)],
                p = 0.5
            ),
            # transforms.RandomGrayscale(p=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.GaussianBlur((5, 5), (1.5, 1.5))],
                p = 0.4
            ),
            # Solarization(0.4),
            # transforms.RandomResizedCrop((200, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.298], std=[0.210]),
        ])
        
        self.data_dir = args.data_dir
        self.image_path = []

        data_info = np.loadtxt(os.path.join(self.data_dir, 'neurobit', 'train.txt'), delimiter=';', dtype=np.str)[1:]
        for info in data_info:
            image_name, _, _ = info.split(',')
            image_path = os.path.join(self.data_dir, 'neurobit', 'image', image_name)

            self.image_path.append(image_path)
        
        data_info = np.loadtxt(os.path.join(self.data_dir, 'neurobit', 'valid.txt'), delimiter=';', dtype=np.str)[1:]
        for info in data_info:
            image_name, _, _ = info.split(',')
            image_path = os.path.join(self.data_dir, 'neurobit', 'image', image_name)

            self.image_path.append(image_path)
        
        data_info = np.loadtxt(os.path.join(self.data_dir, 'neurobit', 'test.txt'), delimiter=';', dtype=np.str)[1:]
        for info in data_info:
            image_name, _, _ = info.split(',')
            image_path = os.path.join(self.data_dir, 'neurobit', 'image', image_name)

            self.image_path.append(image_path)

    def __getitem__(self, index):
        image = Image.open(self.image_path[index]).convert('RGB')
        
        return self.tfm(image), self.tfm(image)
    def __len__(self):
        return len(self.image_path)

