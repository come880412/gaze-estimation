import os
import glob
import cv2
import math
from tqdm import tqdm
from PIL import Image
import argparse
import random
import numpy as np
import json 

'''

Please put the directory containing videos under righteye/ or lefteye/

args.save_root/
    |
    ├─neurobit/ (will be automatically created)
    |   |
    |   ├─image/
    |   |   ├─0000000.png
    |   |   ├─0000001.png
    |   |   ...
    |   ├─image_test/
    |   |   ├─0000000.png
    |   |   ├─0000001.png
    |   |   ...
    |   └─train.txt
    |   └─valid.txt
    |   └─test.txt
    |
args.data_dir/
    ├─YYYYMMDD_H14_NSSxxxxx
    |   ├─0.mp4
    |   ├─1.mp4
    |   ├─2.mp4
    |   ...

video frame size (H, W): (400, 1280) 
'''

def arg_parser():
        
    parser = argparse.ArgumentParser()

    ''' Paths '''
    parser.add_argument('--save_root', type=str, default="../../dataset/neurobit_2")
    parser.add_argument('--data_dir', type=str, default="../../neurobit_data")
    parser.add_argument('--frame_save_num', type=int, default=180)

    ''' Parameters'''
    parser.add_argument('--h', type=float, default=6.8)   
    parser.add_argument('--w', type=float, default=7.7)
    parser.add_argument('--c', type=tuple, default=(0,0)) # (x,y)
    parser.add_argument('--d', type=float, default=78)    # offset
    parser.add_argument('--Lefteye_ROI', type=tuple, default=(0, 400, 640, 1280)) # (left, right, top, bot)
    parser.add_argument('--Righteye_ROI', type=tuple, default=(0, 400, 0, 640))   # (left, right, top, bot)
    
    args = parser.parse_args()

    os.makedirs(os.path.join(args.save_root, "image"), exist_ok=True)

    return args

def get_yaw_pitch(i, h, w, c, d):
    top_left_y = 4 * h + c[1]
    top_left_x = -6 * w + c[0]
    pitch = math.atan( (top_left_y - (i//13) * h) / d) * 180 / math.pi
    yaw   = math.atan( (top_left_x + (i%13)  * w) / d) * 180 / math.pi
    return yaw, pitch

def DataPreprocessing_length(): # length-based
    args = arg_parser()

    Left_files = ['20211210_H14_NSS00000', '20220121_H14_NSS00121', '20220224_H14_NSS00001']
    Right_files = ['20211210_H14_NSS11111', '20220121_H14_NSS00122', '20220224_H14_NSS00002']

    ''' parametrers '''
    L_top, L_bot, L_left, L_right = args.Lefteye_ROI[0], args.Lefteye_ROI[1], args.Lefteye_ROI[2], args.Lefteye_ROI[3]
    R_top, R_bot, R_left, R_right = args.Righteye_ROI[0], args.Righteye_ROI[1], args.Righteye_ROI[2], args.Righteye_ROI[3]

    # Data_split 
    total_files = Left_files + Right_files
    total_file = []
    for i in range(len(total_files)):
        total_file += os.listdir(os.path.join(args.data_dir, total_files[i]))
    total_file = np.array(total_file)

    num_file = len(total_file)
    random_index = np.random.choice(num_file, num_file, replace=False)
    train_index = total_file[random_index[:int(num_file*0.8)]]
    valid_index = total_file[random_index[int(num_file*0.8):int(num_file*0.9)]]
    test_index = total_file[random_index[int(num_file*0.9):]]

    train_data = [['image_name, yaw, pitch']]
    valid_data = [['image_name, yaw, pitch']]
    test_data = [['image_name, yaw, pitch']]
    image_idx = 0

    broken_video = ['20220121_160001_H14_NSS00122_Test1.mp4', '20220121_160007_H14_NSS00122_Test1.mp4', '20220121_160140_H14_NSS00122_Test1.mp4', 
                    '20220121_160520_H14_NSS00122_Test1.mp4', '20220121_160526_H14_NSS00122_Test1.mp4', '20220121_160606_H14_NSS00122_Test1.mp4']
    # left eyes
    for d in Left_files: # iterate thru left_eye video directories
        with open(os.path.join(args.data_dir, d + '.json')) as f:
            data = json.load(f)
        args.d = data['distance_to_grid']

        video_files = sorted(glob.glob(os.path.join(args.data_dir, d,"*")))
        assert(len(video_files) == 9*13)

        for i, v in enumerate(tqdm(video_files)): # each dir represents
            video_name = v.split('/')[-1]
            if video_name in broken_video:
                continue

            yaw,pitch = get_yaw_pitch(i=float(i),h=args.h,w=args.w,c=args.c,d=args.d) # length-based label

            video = cv2.VideoCapture(v)
            success = True
            image_count = 0
            while(success):
                success, frame = video.read() # only take the first frame (or the dataset will be too big)
                if(not success or image_count == args.frame_save_num): break
                if image_count >=30:
                    im = frame[L_top:L_bot, L_left:L_right, :]
                    im = Image.fromarray(im)
                    im.save(os.path.join(args.save_root, "image", f'{str(image_idx).zfill(7)}.png'))
                    
                    if video_name in train_index:
                        train_data.append([f'{str(image_idx).zfill(7)}.png, {str(yaw)},{str(pitch)}'])
                    elif video_name in valid_index:
                        valid_data.append([f'{str(image_idx).zfill(7)}.png, {str(yaw)},{str(pitch)}'])
                    else:
                        test_data.append([f'{str(image_idx).zfill(7)}.png, {str(yaw)},{str(pitch)}'])

                    image_idx += 1
                image_count += 1

    # # right eyes
    for d in Right_files: # iterate thru right_eye video directories
        with open(os.path.join(args.data_dir, d + '.json')) as f:
            data = json.load(f)
        args.d = data['distance_to_grid']

        video_files = sorted(glob.glob(os.path.join(args.data_dir, d ,"*")))
        assert(len(video_files) == 9*13)

        for i, v in enumerate(tqdm(video_files)):
            video_name = v.split('/')[-1]
            if video_name in broken_video:
                continue

            yaw,pitch = get_yaw_pitch(i=float(i),h=args.h,w=args.w,c=args.c,d=args.d) # length-based label

            video = cv2.VideoCapture(v)
            success = True
            image_count = 0
            while(success):
                success, frame = video.read() # only take the first frame (or the dataset will be too big)
                if(not success or image_count == args.frame_save_num): break
                if image_count >=30:
                    im = frame[R_top:R_bot, R_left:R_right, :]
                    im = Image.fromarray(im)
                    im.save(os.path.join(args.save_root, "image", f'{str(image_idx).zfill(7)}.png'))
                
                    if video_name in train_index:
                        train_data.append([f'{str(image_idx).zfill(7)}.png, {str(yaw)},{str(pitch)}'])
                    elif video_name in valid_index:
                        valid_data.append([f'{str(image_idx).zfill(7)}.png, {str(yaw)},{str(pitch)}'])
                    else:
                        test_data.append([f'{str(image_idx).zfill(7)}.png, {str(yaw)},{str(pitch)}'])

                    image_idx += 1
                image_count += 1

    return train_data, valid_data, test_data, image_idx

def DataPreprocessing_angle(train_data, valid_data, test_data, image_idx): # angle-based
    args = arg_parser()

    Left_files = '20211203_H14_NSS00000'
    Right_files = '20211203_H14_NSS11111'

    ''' parametrers '''
    L_top, L_bot, L_left, L_right = args.Lefteye_ROI[0], args.Lefteye_ROI[1], args.Lefteye_ROI[2], args.Lefteye_ROI[3]
    R_top, R_bot, R_left, R_right = args.Righteye_ROI[0], args.Righteye_ROI[1], args.Righteye_ROI[2], args.Righteye_ROI[3]

    Left_dir = os.path.join(args.data_dir, Left_files)
    Right_dir = os.path.join(args.data_dir, Right_files)

    # Data_split 
    total_files = [Left_files, Right_files]
    total_file = []
    for i in range(len(total_files)):
        total_file += os.listdir(os.path.join(args.data_dir, total_files[i]))
    total_file = np.array(total_file)

    num_file = len(total_file)
    random_index = np.random.choice(num_file, num_file, replace=False)
    train_index = total_file[random_index[:int(num_file*0.8)]]
    valid_index = total_file[random_index[int(num_file*0.8):int(num_file*0.9)]]
    test_index = total_file[random_index[int(num_file*0.9):]]

    # left eyes
    video_files = sorted(os.listdir(Left_dir))
    assert(len(video_files) == 9*13)

    for i, video_name in enumerate(tqdm(video_files)): # each dir represents
        video_path = os.path.join(Left_dir, video_name)
        yaw, pitch = -30 + (i%13) * 5, 20 - (i//13) * 5 # angle-based label

        video = cv2.VideoCapture(video_path)
        success = True
        image_count = 0
        while(success):
            success, frame = video.read() # only take the first frame (or the dataset will be too big)
            if(not success or image_count == args.frame_save_num): break
            if image_count >=30:
                im = frame[L_top:L_bot, L_left:L_right, :]
                im = Image.fromarray(im)
                im.save(os.path.join(args.save_root, "image", f'{str(image_idx).zfill(7)}.png'))

                if video_name in train_index:
                    train_data.append([f'{str(image_idx).zfill(7)}.png, {str(yaw)},{str(pitch)}'])
                elif video_name in valid_index:
                    valid_data.append([f'{str(image_idx).zfill(7)}.png, {str(yaw)},{str(pitch)}'])
                else:
                    test_data.append([f'{str(image_idx).zfill(7)}.png, {str(yaw)},{str(pitch)}'])
                image_idx += 1
            image_count += 1

    # # right eyes
    video_files = sorted(os.listdir(Right_dir))
    assert(len(video_files) == 9*13)

    for i, video_name in enumerate(tqdm(video_files)): # each dir represent
        video_path = os.path.join(Right_dir, video_name)
        yaw, pitch = -30 + (i%13) * 5, 20 - (i//13) * 5 # angle-based label

        video = cv2.VideoCapture(video_path)
        success = True
        image_count = 0
        while(success):
            success, frame = video.read() # only take the first frame (or the dataset will be too big)
            if(not success or image_count == args.frame_save_num): break
            if image_count >=30:
                im = frame[R_top:R_bot, R_left:R_right, :]
                im = Image.fromarray(im)
                im.save(os.path.join(args.save_root, "image", f'{str(image_idx).zfill(7)}.png'))
            
                if video_name in train_index:
                    train_data.append([f'{str(image_idx).zfill(7)}.png, {str(yaw)},{str(pitch)}'])
                elif video_name in valid_index:
                    valid_data.append([f'{str(image_idx).zfill(7)}.png, {str(yaw)},{str(pitch)}'])
                else:
                    test_data.append([f'{str(image_idx).zfill(7)}.png, {str(yaw)},{str(pitch)}'])

                image_idx += 1
            image_count += 1
    np.savetxt(os.path.join(args.save_root, 'train.txt'),  train_data, fmt='%s', delimiter=',')
    np.savetxt(os.path.join(args.save_root, 'valid.txt'),  valid_data, fmt='%s', delimiter=',')
    np.savetxt(os.path.join(args.save_root, 'test.txt'),  test_data, fmt='%s', delimiter=',')

    return train_data, valid_data, test_data

if __name__ == "__main__":
    random.seed(2022)
    np.random.seed(2022)
    train_data, valid_data, test_data, image_idx = DataPreprocessing_length()
    DataPreprocessing_angle(train_data, valid_data, test_data, image_idx)

    print('------ Statistics ---------')
    print('Training data: ', len(train_data))
    print('Validation data: ', len(valid_data))
    print('Testing data: ', len(test_data))
    print('---------------------------')
