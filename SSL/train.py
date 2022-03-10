import argparse
import tqdm
import os

import torch
from torchvision import models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from byol_pytorch.byol_pytorch import BYOL
from utils import *
from dataset import Dikablis_data, Neurobit_data

import warnings
warnings.filterwarnings("ignore")

def SSL(args, model, train_loader, val_loader, optimizer, scheduler):
    model.train()

    learner = BYOL(
        model,
        image_size = (args.image_width, args.image_height),
        hidden_layer = 'avgpool',
        projection_size = 256,           # the projection size
        projection_hidden_size = 4096,   # the hidden dimension of the MLP for both the projection and prediction
        moving_average_decay = 0.99      # the moving average decay factor for the target encoder, already set at what paper recommends
    )

    print('Start training!')
    train_iterator = iter(train_loader)
    pbar = tqdm.tqdm(total=args.total_steps, ncols=0, desc="Training", unit=" step")
    for step in range(args.total_steps):
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        try:
            x1, x2 = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            x1, x2 = next(train_iterator)

        x1, x2 = x1.cuda(), x2.cuda()
        loss = learner(x1, x2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.update()
        pbar.set_postfix(
            loss=f"{loss:.4f}",
            lr=f"{lr:.6f}"
        )

        torch.save(model.state_dict(), '%s/model_final.pth' % (args.saved_model))
        scheduler.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-supervised learning")
    parser.add_argument('--data_dir', type=str, default="../../dataset/TEyeD")
    parser.add_argument('--saved_model', type=str, default='./checkpoints/resnet18')
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--model', type=str, default='resnet18', help='resnet18/resnext50')

    ''' paramters '''
    parser.add_argument('--image_width', type=int, default=320, help='Image width')
    parser.add_argument('--image_height', type=int, default=200, help='Image height')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--total_steps', type=int, default=100000)
    parser.add_argument('--decay_steps', type=int, default=20000)

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)

    args = parser.parse_args()
    os.makedirs(args.saved_model, exist_ok=True)
    
    Set_seed(args.seed)
    
    train_data = Dikablis_data(args, 'train')
    valid_data = Dikablis_data(args, 'valid')

    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False, shuffle=False)
    print('Number of training data : ', len(train_data))
    print('Number of validation data : ', len(valid_data))
    
    if args.model == 'resnext50':
        model = models.resnext50_32x4d(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif args.model == 'resnet18':
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
    print('Using the model of %s' % (args.model))

    if args.load:
        print('Load pretrained model!!')
        model.load_state_dict(torch.load(args.load))
        
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1.5e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_steps, gamma=0.5)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.decay_steps, eta_min=1e-5)
    model = model.cuda()
    SSL(args, model, train_loader, valid_loader, optimizer, scheduler)

    

    