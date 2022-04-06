import cv2
import numpy as np
from tqdm import tqdm

import os
import argparse
import random

import warnings
warnings.filterwarnings("ignore")

'''
Only process on Dikablis dataset

output directory tree will looks like this:

args.root/
        ├─TEyeDS (Original dataset)
        |   └─Dikablis
        |       ├─ANNOTATIONS
        |       ├─VIDEOS
        |       ├─Readme.txt
        |
        └─TEyeD/ (Will create this directory automatically)
        └─TEyeD_valid/ (Will create this directory automatically)
'''

np.random.seed(2022)
random.seed(2022)
def valid_split(root): # For closed_eyes and open_eyes
    data_path = root

    open_eye_path_list = []
    closed_eye_path_list = []

    open_eye_folder_list = os.listdir(os.path.join(data_path, 'TEyeD'))
    for video_name in open_eye_folder_list:
        if video_name == 'train.txt' or video_name == 'valid.txt':
            continue
        open_eye_list = os.listdir(os.path.join(data_path, 'TEyeD', video_name, 'frame'))

        for frame_name in open_eye_list:
            open_eye_path_list.append([os.path.join(data_path, 'TEyeD', video_name, 'frame', frame_name), '1'])

    closed_eye_list = os.listdir(os.path.join(data_path, 'TEyeD_valid', 'frame'))
    for frame_name in closed_eye_list:
        closed_eye_path_list.append([os.path.join(data_path, 'TEyeD_valid', 'frame', frame_name), '0'])
    
    random.shuffle(open_eye_path_list)
    random.shuffle(closed_eye_path_list)
    open_eye_path_list = open_eye_path_list[:len(closed_eye_path_list)]

    open_eye_path_list = np.array(open_eye_path_list)
    closed_eye_path_list = np.array(closed_eye_path_list)

    num_open = len(open_eye_path_list)
    num_close = len(closed_eye_path_list)

    train_data = np.concatenate((open_eye_path_list[:int(num_open*0.8)], closed_eye_path_list[:int(num_close * 0.8)]))
    valid_data = np.concatenate((open_eye_path_list[int(num_open*0.8):int(num_open*0.9)], closed_eye_path_list[int(num_close*0.8):int(num_close * 0.9)]))
    test_data = np.concatenate((open_eye_path_list[int(num_open*0.9):], closed_eye_path_list[int(num_close * 0.9):]))

    print('------Statistics_valid--------')
    print('num_training data: ', train_data.shape)
    print('num_validation data: ', valid_data.shape)
    print('num_testing data: ', test_data.shape)

    np.savetxt(os.path.join(data_path, 'TEyeD_valid', 'train.txt'),  train_data, fmt='%s', delimiter=',')
    np.savetxt(os.path.join(data_path, 'TEyeD_valid', 'valid.txt'),  valid_data, fmt='%s', delimiter=',')
    np.savetxt(os.path.join(data_path, 'TEyeD_valid', 'test.txt'),  test_data, fmt='%s', delimiter=',')

def data_process(args, video_path, label_path):
    train_val_ratio = args.train_val_ratio
    root = args.root

    video_list = os.listdir(video_path).sort()
    train_val_video = []

    tqdm.write(f'Number of videos: {len(video_list)}')
    save_path = os.path.join(root, "TEyeD")
    save_valid_path = os.path.join(root, 'TEyeD_valid', "frame")

    os.makedirs(save_valid_path, exist_ok=True)
    valid_count = 0

    # BROKEN FILE
    broken = []
    broken.append("DikablisSS_10_1.mp4")
    with open("./others/pupil_seg_broken.txt", 'r') as p:
        with open("./others/iris_seg_broken.txt", 'r') as i:
            with open("./others/lid_seg_broken.txt", 'r') as l:
                for line in p.readlines():
                    broken.append(line.strip())
                for line in i.readlines():
                    broken.append(line.strip())
                for line in l.readlines():
                    broken.append(line.strip())

    for video_name in tqdm(video_list):
        if video_name in broken:
            continue    
        gaze_info = np.loadtxt(os.path.join(label_path,f'{video_name}gaze_vec.txt'), delimiter=';', dtype=np.str)[1:]
        pupil_info = np.loadtxt(os.path.join(label_path,f'{video_name}pupil_lm_2D.txt'), delimiter=';', dtype=np.str)[1:]
        iris_info = np.loadtxt(os.path.join(label_path,f'{video_name}iris_lm_2D.txt'), delimiter=';', dtype=np.str)[1:]
        lid_info = np.loadtxt(os.path.join(label_path,f'{video_name}lid_lm_2D.txt'), delimiter=';', dtype=np.str)[1:]
        '''
        image shape: (288, 384, 3) in Dikablis
        '''

        # Source video
        video = cv2.VideoCapture(os.path.join(video_path, video_name))
        success = True
        source_image_count = 0
        source_image_idx = 0

        while(success):
            # OUTPUT DIR
            os.makedirs(os.path.join(save_path, video_name[:-4], "frame"), exist_ok=True)
            os.makedirs(os.path.join(save_path, video_name[:-4], "visualization"), exist_ok=True)
            os.makedirs(os.path.join(save_path, video_name[:-4], "iris_landmark"), exist_ok=True)
            os.makedirs(os.path.join(save_path, video_name[:-4], "pupil_landmark"), exist_ok=True)
            os.makedirs(os.path.join(save_path, video_name[:-4], "lid_landmark"), exist_ok=True)
            os.makedirs(os.path.join(save_path, video_name[:-4], "gaze_vector"), exist_ok=True)
            os.makedirs(os.path.join(save_path, video_name[:-4], "center"), exist_ok=True)

            success, frame = video.read()
            if not success:
                break
            gaze_vector = gaze_info[source_image_idx, 1:-1].astype(float)
            iris_landmark = iris_info[source_image_idx, 2:-1].astype(float)
            lid_landmark = lid_info[source_image_idx, 2:-1].astype(float)
            pupil_landmark = pupil_info[source_image_idx, 2:-1].astype(float)

            center_x = 0.
            center_y = 0.
            for i in range(0, len(pupil_landmark) // 2, 1):
                center_x += pupil_landmark[i * 2]
                center_y += pupil_landmark[i*2+1]
            center_x /= (len(pupil_landmark) / 2)
            center_y /= (len(pupil_landmark) / 2)
            center = [center_x, center_y]

            if gaze_vector[0] == -1 and gaze_vector[1] == -1 and gaze_vector[2] == -1: # closed eyes
                cv2.imwrite(os.path.join(save_valid_path,f'{str(valid_count).zfill(7)}.png'), frame)
                source_image_idx += 1
                valid_count += 1
                continue
            if sum(iris_landmark<0) >= 1 or sum(lid_landmark<0) >=1 or sum(pupil_landmark<0) >=1: # if landmark is negative
                source_image_idx += 1
                continue
            landmark_all = np.concatenate((iris_landmark, lid_landmark, pupil_landmark))
    
            image_landmark = frame.copy()
            for i in range(0, len(landmark_all), 2):
                x, y = landmark_all[i], landmark_all[i+1]
                image_landmark = cv2.circle(image_landmark, (int(x),int(y)), 1, (0,0,255), 3)

            cv2.imwrite(os.path.join(save_path, video_name[:-4], "frame",f'{str(source_image_count).zfill(7)}.png'), frame)
            cv2.imwrite(os.path.join(save_path, video_name[:-4], "visualization",f'{str(source_image_count).zfill(7)}.png'), image_landmark)

            np.savetxt(os.path.join(save_path, video_name[:-4], "gaze_vector", f'{str(source_image_count).zfill(7)}.txt'),  gaze_vector, fmt='%s', delimiter=',')
            np.savetxt(os.path.join(save_path, video_name[:-4], "center", f'{str(source_image_count).zfill(7)}.txt'),  center, fmt='%s', delimiter=',')
            np.savetxt(os.path.join(save_path, video_name[:-4], "iris_landmark", f'{str(source_image_count).zfill(7)}.txt'),  iris_landmark, fmt='%s', delimiter=',')
            np.savetxt(os.path.join(save_path, video_name[:-4], "lid_landmark", f'{str(source_image_count).zfill(7)}.txt'),  lid_landmark, fmt='%s', delimiter=',')
            np.savetxt(os.path.join(save_path, video_name[:-4], "pupil_landmark", f'{str(source_image_count).zfill(7)}.txt'),  pupil_landmark, fmt='%s', delimiter=',')
            source_image_count += 1
            source_image_idx += 1

            if source_image_count == args.fpv: # reach desired frames per video
                break
        
        train_val_video.append(video_name[:-4])

    train_val_video = np.array(train_val_video)
    len_video = len(train_val_video)
    random_num_list = np.random.choice(len_video, len_video, replace=False)

    train_index = np.array(random_num_list[:int(len(random_num_list) * train_val_ratio)], dtype=int)
    val_index = np.array(random_num_list[int(len(random_num_list) * train_val_ratio):], dtype=int)

    train_video_list = train_val_video[train_index]
    val_video_list = train_val_video[val_index]

    np.savetxt(os.path.join(save_path, 'train.txt'),  train_video_list, fmt='%s', delimiter=',')
    np.savetxt(os.path.join(save_path, 'valid.txt'),  val_video_list, fmt='%s', delimiter=',')

    print('------Statistics_gaze--------')
    print('num_training data: ', len(train_video_list))
    print('num_validation data: ', len(val_video_list))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../../dataset/', help='Root directory')
    parser.add_argument('--fpv', type=int, default=1500, help='How many frames per video you want to store.')
    parser.add_argument('--train_val_ratio', type=float, default=0.99, help='Training and validation ratio')
    args = parser.parse_args()

    # PATH
    video_path = os.path.join(args.root, "TEyeDS/Dikablis/VIDEOS")
    label_path = os.path.join(args.root, "TEyeDS/Dikablis/ANNOTATIONS")

    data_process(args, video_path, label_path)
    valid_split(args.root)