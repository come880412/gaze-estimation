import os
import numpy as np
import random

def valid_split(root):
    np.random.seed(2022)
    random.seed(2022)

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


    
