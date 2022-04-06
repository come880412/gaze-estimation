import os
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random

from utils import vec2angle

class Dikablis_data_center(data.Dataset):

    def __init__(self, args, mode):
        self.in_w, self.in_h = 384, 288 # Width and height of Dikablis dataset 
        self.out_w, self.out_h = args.image_width, args.image_height
        self.ratio_w, self.ratio_h = self.out_w / self.in_w, self.out_h / self.in_h

        self.tfm = transforms.Compose([
                    transforms.Resize((self.out_h, self.out_w)), # Since the width and the height of cv2 is (height, width)
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                    # transforms.GaussianBlur((5, 5), (1.5, 1.5)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.298], std=[0.210])
                ])
        
        self.data_dir = args.data_dir
        self.mode = mode

        video_name_file = np.loadtxt(os.path.join(self.data_dir, '%s.txt' % self.mode), delimiter=';', dtype=np.str)
        
        self.image_path = []
        self.center_path = []
        for video_name in video_name_file:
            image_path = os.path.join(self.data_dir, video_name, 'frame')

            image_frame_list = sorted(os.listdir(image_path))
            for frame_name in image_frame_list:
                frame_path = os.path.join(image_path, frame_name)
                center_path = os.path.join(self.data_dir, video_name, 'center', frame_name[:-4] + '.txt')

                self.image_path.append(frame_path)
                self.center_path.append(center_path)
        assert len(self.image_path) == len(self.center_path)

    def __getitem__(self, index):
        image = Image.open(self.image_path[index])
        center = np.loadtxt(self.center_path[index], delimiter=',', dtype=np.float)
        center[0] = center[0] * self.ratio_w
        center[1] = center[1] * self.ratio_h
        
        return self.tfm(image),  torch.FloatTensor(center), self.image_path[index]

    def __len__(self):
        return len(self.image_path)

class Dikablis_data_valid(data.Dataset):

    def __init__(self, args, mode):
        self.in_w, self.in_h = 384, 288 # Width and height of Dikablis dataset 
        self.out_w, self.out_h = args.image_width, args.image_height

        self.train_tfm = transforms.Compose([
                    transforms.Resize((self.out_h, self.out_w)), # Since the width and the height of cv2 is (height, width)
                    transforms.RandomHorizontalFlip(p=0.3),
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                    # transforms.GaussianBlur((5, 5), (1.5, 1.5)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.298], std=[0.210])
                ])
        self.test_tfm = transforms.Compose([
                    transforms.Resize((self.out_h, self.out_w)), # Since the width and the height of cv2 is (height, width)
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.298], std=[0.210])
                ])
        
        self.data_dir = args.data_dir
        self.mode = mode

        self.data_info = np.loadtxt(os.path.join(self.data_dir, '%s.txt' % self.mode), delimiter=',', dtype=np.str)

    def __getitem__(self, index):
        image_path, label = self.data_info[index]
        label = np.array(label).astype(np.float)
        image = Image.open(image_path).convert('RGB')
        if self.mode == 'valid' or self.mode == 'test':
            image = self.test_tfm(image)
        else:
            image = self.train_tfm(image)

        return image,  torch.FloatTensor(label)

    def __len__(self):
        return len(self.data_info)

class Dikablis_data_gaze(data.Dataset):

    def __init__(self, args, mode):
        self.in_w, self.in_h = 384, 288 # Width and height of Dikablis dataset 
        self.out_w, self.out_h = args.image_width, args.image_height

        self.tfm = transforms.Compose([
                    transforms.Resize((self.out_h, self.out_w)), # Since the width and the height of cv2 is (height, width)
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                    # transforms.GaussianBlur((5, 5), (1.5, 1.5)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.298], std=[0.210])
                ])
        
        self.data_dir = args.data_dir
        self.mode = mode

        video_name_file = np.loadtxt(os.path.join(self.data_dir, '%s.txt' % self.mode), delimiter=';', dtype=np.str)
        
        self.image_path = []
        self.gaze_vector_path = []
        for video_name in video_name_file:
            image_path = os.path.join(self.data_dir, video_name, 'frame')

            image_frame_list = sorted(os.listdir(image_path))
            for frame_name in image_frame_list:
                frame_path = os.path.join(image_path, frame_name)
                gaze_path = os.path.join(self.data_dir, video_name, 'gaze_vector', frame_name[:-4] + '.txt')

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

class Neurobit_data_center(data.Dataset):

    def __init__(self, args, mode):
        self.in_w, self.in_h = 430, 400 # Width and height of Neurobit dataset 
        self.out_w, self.out_h = args.image_width, args.image_height
        self.ratio_w, self.ratio_h = self.out_w / self.in_w, self.out_h / self.in_h

        self.tfm = transforms.Compose([
                    transforms.Resize((self.out_h, self.out_w)), # Since the width and the height of cv2 is (height, width)
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.298], std=[0.210])
                ])
        
        self.data_dir = args.data_dir
        self.mode = mode

        file_info = np.loadtxt(os.path.join(self.data_dir, '%s_center.txt' % self.mode), delimiter=',', dtype=np.str)[1:]

        self.image_path = []
        self.center = []
        for info in file_info:
            frame_name, center_x, center_y = info

            image_path = os.path.join(self.data_dir, 'image', frame_name)

            self.image_path.append(image_path)
            self.center.append([float(center_x), float(center_y)])
        assert len(self.image_path) == len(self.center)

    def __getitem__(self, index):
        image = Image.open(self.image_path[index])
        center_x, center_y = self.center[index]

        center_x = center_x * self.ratio_w
        center_y = center_y * self.ratio_h

        center = np.array([center_x, center_y])
        
        return self.tfm(image),  torch.FloatTensor(center), self.image_path[index]

    def __len__(self):
        return len(self.image_path)

class Neurobit_data_valid(data.Dataset):
    def __init__(self, args, mode):
        self.in_w, self.in_h = 640, 400 # Width and height of Dikablis dataset 
        self.out_w, self.out_h = args.image_width, args.image_height

        self.train_tfm = transforms.Compose([
                    transforms.Resize((self.out_h, self.out_w)), # Since the width and the height of cv2 is (height, width)
                    transforms.RandomHorizontalFlip(p=0.3),
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.298], std=[0.210])
                ])
        self.test_tfm = transforms.Compose([
                    transforms.Resize((self.out_h, self.out_w)), # Since the width and the height of cv2 is (height, width)
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.298], std=[0.210])
                ])
        
        self.data_dir = args.data_dir
        self.mode = mode
        self.image_path = []

        self.data_info = np.loadtxt(os.path.join(self.data_dir, '%s.txt' % self.mode), delimiter=',', dtype=np.str)
        for info in self.data_info:
            self.image_path.append(os.path.join(self.data_dir, info[0]))


    def __getitem__(self, index):
        image_path = self.image_path[index]
        label = self.data_info[index][1]
        label = np.array(label).astype(np.float)
        image = Image.open(image_path).convert('RGB')
        if self.mode == 'valid' or self.mode == 'test':
            # image = image.crop((50, 110, 400, 310))
            image = self.test_tfm(image)
        else:
            # vertical = random.randint(-15, 15)
            # horizontal = random.randint(-15, 15)
            # image = image.crop((50+horizontal, 110+vertical, 400+horizontal, 310+vertical))
            image = self.train_tfm(image)

        return image,  torch.FloatTensor(label)

    def __len__(self):
        return len(self.data_info)

class Neurobit_data_gaze(data.Dataset):
    def __init__(self, args, mode):
        self.out_w, self.out_h = args.image_width, args.image_height
        self.train_tfm = transforms.Compose([
                    transforms.Resize((self.out_h, self.out_w)),
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.298], std=[0.210])
                ])
        self.test_tfm = transforms.Compose([
                    transforms.Resize((self.out_h, self.out_w)),
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

            if os.path.isfile(image_path):
                self.image_path.append(image_path)
                self.gaze_vector.append([float(yaw), float(pitch)])
        assert len(self.image_path) == len(self.gaze_vector)
    
    def __getitem__(self, index):
        image = Image.open(self.image_path[index])

        if self.mode == 'valid' or self.mode == 'test':
            image = image.crop((50, 110, 400, 310)) # for left_eyes
            # image = image.crop((0, 110, 350, 310)) # for right_eyes
            image = self.test_tfm(image)
        else:
            """for left_eyes"""
            vertical = random.randint(-15, 15)
            horizontal = random.randint(-15, 15)
            image = image.crop((50+horizontal, 110+vertical, 400+horizontal, 310+vertical))
            """for right_eyes"""
            # vertical = random.randint(-50, 50)
            # horizontal = random.randint(0, 50)
            # image = image.crop((0+horizontal, 110+vertical, 350+horizontal, 310+vertical))

            image = self.train_tfm(image)
        
        return image, torch.FloatTensor(self.gaze_vector[index]), self.image_path[index]
    
    def __len__(self):
        return len(self.image_path)


