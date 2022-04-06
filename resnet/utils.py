import random
import numpy as np
import cv2

import torch

def visualize_center(args, image_path, pred, true_center, save_path, idx):
    pred = pred.cpu().detach().numpy()
    true_center = true_center.cpu().detach().numpy()

    idx = str(idx).zfill(4)
    image = cv2.imread(image_path)

    in_w, in_h = args.image_width, args.image_height
    out_h, out_w, _ = image.shape 
    scale_w, scale_h = out_w / in_w, out_h / in_h
    
    cv2.circle(image, (int(round(pred[0] * scale_w)),int(round(pred[1] * scale_h))), 3, (0,0,255), thickness=-1)
    cv2.circle(image, (int(round(true_center[0] * scale_w)),int(round(true_center[1] * scale_h))), 3, (0,255,0), thickness=-1)
    cv2.imwrite('./%s/%s.png' % (save_path, idx), image)

def Set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def vec2angle(gaze):
    '''
    gaze shape : (x, y, z)
    
    * from your viewpoint *
    x+: right
    y+: down
    z+: point to you
    pitch+: up
    yaw+: left(your pov) ; right (patient's pov)
    '''
    x, y, z = gaze[0], gaze[1], gaze[2]
    pitch = - np.arctan(y/z) * 180 / np.pi
    yaw = - np.arctan(x/z) * 180 / np.pi
    return np.array([pitch, yaw])[np.newaxis,:]

def get_acc(y_pred, y_true, correct, total, threshold=0.9):
        """ ACC metric
        y_pred: the predicted score of each class, shape: (Batch_size, num_classes)
        y_true: the ground truth labels, shape: (Batch_size,) for 'multi-class' or (Batch_size, n_classes) for 'multi-label'
        """
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()

        for i in range(y_true.shape[0]):
            if y_pred[i] >= threshold:
                pred = 1
            else:
                pred = 0
            label = y_true[i]
            total[int(label)] += 1

            if pred == label:
                correct[int(pred)] += 1
        
        return correct, total
