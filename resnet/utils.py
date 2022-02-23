import random
import numpy as np

import torch

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
