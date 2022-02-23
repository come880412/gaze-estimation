import argparse
import os
import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from utils import *
from dataset import Dikablis_data, Neurobit_data
import torchvision.models as models

import warnings
warnings.filterwarnings("ignore")

torch.autograd.set_detect_anomaly(True)
def valid(args, model, valid_loader, step):
    model.eval()
    pbar = tqdm.tqdm(total=len(valid_loader), ncols=0, desc="Valid[%d/%d]"%(step, args.total_steps), unit=" step")

    total_loss = 0
    yaw_total_loss = 0
    pitch_total_loss = 0
    total_num_image = 0
    criterion = nn.MSELoss(reduction='none')
    with torch.no_grad():
        for idx, (image, gaze) in enumerate(valid_loader):
            image = image.cuda()
            gaze = gaze.cuda()

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
        return total_loss / total_num_image

def train(image, gaze, model, optimizer):
    criterion = nn.MSELoss(reduction='none')
    model.train()

    # statistics
    image = image.cuda()
    gaze = gaze.cuda()

    optimizer.zero_grad()
    pred = model(image)

    loss = criterion(pred, gaze)
    yaw_loss, pitch_loss = loss[:, 0], loss[:, 1]

    yaw_loss = torch.mean(yaw_loss)
    pitch_loss = torch.mean(pitch_loss)
    train_loss = torch.mean(loss)

    train_loss.backward()
    optimizer.step()

    return train_loss, yaw_loss, pitch_loss

def main(args, model, train_loader, valid_loader, optimizer, scheduler):

    save_name = os.path.join(args.saved_model, 'model_best.pth')
    print('start tarining...')

    min_loss = 100000000000000000
    train_iterator = iter(train_loader)
    pbar = tqdm.tqdm(total=args.total_steps, ncols=0, desc="Training", unit=" step")
    for step in range(args.total_steps):
        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        try:
            image, gaze = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            image, gaze = next(train_iterator)

        train_loss, yaw_loss, pitch_loss = train(image, gaze, model, optimizer)

        pbar.update()
        pbar.set_postfix(
            yaw_loss = f"{yaw_loss:.4f}",
            pitch_loss = f"{pitch_loss:.4f}",
            train_loss = f"{train_loss:.4f}",
            lr=f"{lr:.6f}"
        )

        if (step + 1) % 2000 == 0:
            pbar.close()
            valid_loss = valid(args, model, valid_loader, step +1)

            if valid_loss <= min_loss:
                min_loss = valid_loss
                torch.save(model.state_dict(), save_name)

            pbar = tqdm.tqdm(total=args.total_steps, ncols=0, desc="Training", unit=" step")
            pbar.update(step + 1)
        scheduler.step()
    print('minimum loss: ', min_loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ''' Paths '''
    # parser.add_argument('--data_dir', type=str, default="/home/brianw0924/hdd/TEyeD")
    parser.add_argument('--data_dir', type=str, default="../neurobit/dataset_nocrop")
    parser.add_argument('--dataset', type=str, default="neurobit", choices=["neurobit", "TEyeD"]) # if you set Neurobit, will do random cropping
    parser.add_argument('--saved_model', type=str, default='./checkpoints/gaze_model')
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--model', type=str, default='resnext50', help='resnet18/resnext50')

    ''' paramters '''
    parser.add_argument('--image_width', type=int, default=640, help='Image width')
    parser.add_argument('--image_height', type=int, default=400, help='Image height')
    parser.add_argument('--warm_up', type=int, default=1000, help='Warmup step')
    parser.add_argument('--decay_step', type=int, default=15000, help='')
    parser.add_argument('--decay', type=float, default=0.5, help='')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--total_steps', type=int, default=70000)
    
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)

    args = parser.parse_args()
    os.makedirs(args.saved_model, exist_ok=True)
    Set_seed(args.seed)
    
    if args.dataset == 'TEyeD':
        train_data = Dikablis_data(args, 'train')
        valid_data = Dikablis_data(args, 'valid')
    elif args.dataset == 'neurobit':
        train_data = Neurobit_data(args, 'train')
        valid_data = Neurobit_data(args, 'valid')

    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False, shuffle=False)
    print('Number of training data : ', len(train_data))
    print('Number of validation data : ', len(valid_data))
    
    if args.model == 'resnext50':
        model = models.resnext50_32x4d(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif args.model == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)

    if args.load:
        model.load_state_dict(torch.load(args.load))
        
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR( 
                    optimizer, 
                    step_size=args.decay_step, 
                    gamma=args.decay
                )
    model = model.cuda()

    main(args, model, train_loader, valid_loader, optimizer, scheduler)