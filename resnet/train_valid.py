import argparse
import os
from webbrowser import get
import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from utils import *
from dataset import Dikablis_data_valid
import torchvision.models as models

import warnings
warnings.filterwarnings("ignore")

torch.autograd.set_detect_anomaly(True)
def valid(args, model, valid_loader, epoch):
    model.eval()
    pbar = tqdm.tqdm(total=len(valid_loader), ncols=0, desc="Valid[%d/%d]"%(epoch, args.n_epochs), unit=" step")

    total_loss = 0
    eps = 1e-7
    correct = [0, 0]
    total = [0, 0]
    criterion = nn.BCELoss()
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        for idx, (image, label) in enumerate(valid_loader):
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
                valid_open_acc = f"{(correct[1]/(total[1] + eps))*100:.2f}%",
                valid_close_acc = f"{(correct[0]/(total[0] + eps))*100:.2f}%",
                total_loss = f"{total_loss:.4f}"
            )
        valid_acc = (sum(correct)/sum(total)) * 100
        pbar.close()
        return total_loss, valid_acc

def train(image, label, model, optimizer):
    criterion = nn.BCELoss()
    sigmoid = nn.Sigmoid()

    optimizer.zero_grad()
    pred = model(image)
    pred = sigmoid(pred)
    pred = pred.squeeze()

    loss = criterion(pred, label)

    loss.backward()
    optimizer.step()

    return pred, loss

def main(args, model, train_loader, valid_loader, optimizer, scheduler):

    save_name = os.path.join(args.saved_model, 'model_best.pth')
    print('start tarining...')

    max_acc = 0.
    eps = 1e-7
    
    for epoch in range(args.n_epochs):
        pbar = tqdm.tqdm(total=len(train_loader), ncols=0, desc="Training[%s/%s]" % (epoch, args.n_epochs), unit=" step")
        model.train()
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        
        correct = [0, 0]
        total = [0, 0]
        for image, label in train_loader:
            label = label.squeeze()
            image, label = image.cuda(), label.cuda()

            pred, train_loss = train(image, label, model, optimizer)
            correct, total = get_acc(pred, label, correct, total, args.threshold)

            pbar.update()
            pbar.set_postfix(
                train_loss = f"{train_loss:.4f}",
                train_open_acc = f"{(correct[1]/(total[1] + eps))*100:.2f}%",
                train_close_acc = f"{(correct[0]/(total[0] + eps))*100:.2f}%",
                lr=f"{lr:.6f}"
            )

        pbar.close()
        valid_loss, valid_acc = valid(args, model, valid_loader, epoch)

        if valid_acc >= max_acc:
            max_acc = valid_acc
            torch.save(model.state_dict(), save_name)

        scheduler.step()
    print('max_acc: ', max_acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ''' Paths '''
    # parser.add_argument('--data_dir', type=str, default="/home/brianw0924/hdd/TEyeD")
    parser.add_argument('--data_dir', type=str, default="../../dataset/TEyeD_valid")
    parser.add_argument('--saved_model', type=str, default='./checkpoints/resnet18_valid_SSL')
    parser.add_argument('--load', type=str, default='./checkpoints/SSL_pretrained.pth')
    parser.add_argument('--model', type=str, default='resnet18', help='resnet18/resnext50')

    ''' paramters '''
    parser.add_argument('--image_width', type=int, default=320, help='Image width')
    parser.add_argument('--image_height', type=int, default=200, help='Image height')
    parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--threshold', type=float, default=0.5, help='determine whether eyes are open')
    parser.add_argument('--seed', type=int, default=17)
    
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)

    args = parser.parse_args()
    os.makedirs(args.saved_model, exist_ok=True)
    Set_seed(args.seed)
    
    train_data = Dikablis_data_valid(args, 'train')
    valid_data = Dikablis_data_valid(args, 'valid')

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
    model.fc = nn.Linear(model.fc.in_features, 1)
        
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR( 
                    optimizer, 
                    T_max= args.n_epochs,
                    eta_min=1e-5)
    model = model.cuda()

    main(args, model, train_loader, valid_loader, optimizer, scheduler)