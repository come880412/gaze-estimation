import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from utils import *
from dataset import Neurobit_data
import torchvision.models as models

import warnings
warnings.filterwarnings("ignore")

def test(args, model, test_loader, mode):
    pbar = tqdm.tqdm(total=len(test_loader), ncols=0, desc="%s" % mode, unit=" step")
    criterion = nn.MSELoss(reduction='none')

    total_loss = 0
    yaw_total_loss = 0
    pitch_total_loss = 0
    total_num_image = 0
    with torch.no_grad():
        for image, gaze in test_loader:
            image = image.float().cuda()
            gaze = gaze.cuda() # (yaw, pitch)

            pred = model(image)

            loss = criterion(pred, gaze)
            yaw_loss, pitch_loss = loss[:, 0], loss[:, 1]

            total_num_image += len(image)
            yaw_total_loss += torch.sum(yaw_loss)
            pitch_total_loss += torch.sum(pitch_loss)
            total_loss += torch.sum(loss)

            pbar.update()
            pbar.set_postfix(
                yaw_loss = f"{yaw_total_loss / total_num_image:.4f}",
                pitch_loss = f"{pitch_total_loss / total_num_image:.4f}",
                total_loss = f"{total_loss / total_num_image:.4f}"
            )
    pbar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ''' Paths '''
    # parser.add_argument('--data_dir', type=str, default="/home/brianw0924/hdd/TEyeD")
    parser.add_argument('--data_dir', type=str, default="../../dataset/neurobit_2")
    parser.add_argument('--load', type=str, default='./checkpoints/resnet18_alldata/model_2.2087.pth')
    parser.add_argument('--mode', type=str, default='test', help='valid/test')

    ''' paramters '''
    parser.add_argument('--image_width', type=int, default=400, help='Image width')
    parser.add_argument('--image_height', type=int, default=300, help='Image height')
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    Set_seed(args.seed)

    test_data = Neurobit_data(args, args.mode)
    test_loader = DataLoader(test_data, batch_size=4, num_workers=4, drop_last=False, shuffle=True)
    
    model = models.resnet18(pretrained=True)
    # model = models.resnext50_32x4d(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    if args.load:
        print('Load model!!')
        model.load_state_dict(torch.load(args.load))
    model.eval()
    model = model.cuda()
    
    test(args, model, test_loader, args.mode)


    


