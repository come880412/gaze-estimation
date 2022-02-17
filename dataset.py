import os
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from utils import get_heatmaps, vec2angle

class Gaze_dataset(data.Dataset):

    def __init__(self, args, mode):
        self.in_w, self.in_h = 384, 288 # Width and height of Dikablis dataset 
        self.out_w, self.out_h = args.image_width, args.image_height
    
        self.heatmap_w = int(self.out_w / 2)
        self.heatmap_h = int(self.out_h / 2)

        self.scale_w, self.scale_h = self.heatmap_w / self.in_w, self.heatmap_h / self.in_h

        self.tfm = transforms.Compose([
                    transforms.Resize((self.out_h, self.out_w)), # Since the width and the height of cv2 is (height, width)
                    transforms.ToTensor(),
                ])
        
        self.data_dir = args.data_dir
        self.dataset = args.dataset
        self.mode = mode

        video_name_file = np.loadtxt(os.path.join(self.data_dir, self.dataset, '%s.txt' % self.mode), delimiter=';', dtype=np.str)
        
        self.image_path = []
        self.lid_landmark_path = []
        self.pupil_landmark_path = []
        self.gaze_vector_path = []
        for video_name in video_name_file:
            image_path = os.path.join(self.data_dir, self.dataset, video_name, 'frame')

            image_frame_list = sorted(os.listdir(image_path))
            for idx, frame_name in enumerate(image_frame_list):
                frame_path = os.path.join(image_path, frame_name)
                gaze_path = os.path.join(self.data_dir, self.dataset, video_name, 'gaze_vector', frame_name[:-4] + '.txt')
                lid_landmark_path = os.path.join(self.data_dir, self.dataset, video_name, 'lid_landmark', frame_name[:-4] + '.txt')
                pupil_landmark_path = os.path.join(self.data_dir, self.dataset, video_name, 'pupil_landmark', frame_name[:-4] + '.txt')

                self.image_path.append(frame_path)
                self.lid_landmark_path.append(lid_landmark_path)
                self.pupil_landmark_path.append(pupil_landmark_path)
                self.gaze_vector_path.append(gaze_path)
        assert len(self.image_path) == len(self.lid_landmark_path) == len(self.pupil_landmark_path) == len(self.gaze_vector_path)

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

        heatmaps = get_heatmaps(w=self.heatmap_w, h=self.heatmap_h, landmarks=landmarks) # (25, heatmap_h, heatmap_w)

        label_info = np.concatenate((gaze, landmarks)).tolist() # 0:gaze, 1~17:lid, 18~25:pupil
        return label_info, heatmaps
        
if __name__ == '__main__':
    dataset = Gaze_dataset('TEyeD', '../TEyeD', 'train')