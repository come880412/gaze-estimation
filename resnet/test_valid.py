import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from utils import *
from dataset import Dikablis_data_valid
import torchvision.models as models

import warnings
warnings.filterwarnings("ignore")

def test(args, model, test_loader, mode):
    model.eval()
    pbar = tqdm.tqdm(total=len(test_loader), ncols=0, desc="%s" % mode, unit=" step")

    total_loss = 0
    eps = 1e-7
    correct = [0, 0]
    total = [0, 0]
    criterion = nn.BCELoss()
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        for idx, (image, label) in enumerate(test_loader):
            image = image.cuda()
            label = label.cuda()

            pred = model(image)
            pred = sigmoid(pred)
            pred = pred.squeeze()

            loss = criterion(pred, label)
            correct, total = get_acc(pred, label, correct, total, args.threshold)

            total_loss += loss

            pbar.update()
            pbar.set_postfix(
                open_acc = f"{(correct[1]/(total[1] + eps))*100:.2f}%",
                close_acc = f"{(correct[0]/(total[0] + eps))*100:.2f}%",
                valid_acc = f"{(sum(correct)/sum(total)) * 100:.2f}%",
                total_loss = f"{total_loss:.4f}"
            )
        pbar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ''' Paths '''
    # parser.add_argument('--data_dir', type=str, default="/home/brianw0924/hdd/TEyeD")
    parser.add_argument('--data_dir', type=str, default="../../dataset/TEyeD_valid")
    parser.add_argument('--load', type=str, default='./checkpoints/resnet18_valid_SSL/model_best.pth')
    parser.add_argument('--mode', type=str, default='valid', help='valid/test')

    ''' paramters '''
    parser.add_argument('--image_width', type=int, default=320, help='Image width')
    parser.add_argument('--image_height', type=int, default=200, help='Image height')
    parser.add_argument('--threshold', type=float, default=0.5, help='determine whether eyes are open')
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    Set_seed(args.seed)

    test_data = Dikablis_data_valid(args, args.mode)
    test_loader = DataLoader(test_data, batch_size=128, num_workers=4, drop_last=False, shuffle=True)
    
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    if args.load:
        print('Load model!!')
        model.load_state_dict(torch.load(args.load))
    model.eval()
    model = model.cuda()
    
    test(args, model, test_loader, args.mode)
