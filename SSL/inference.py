'''
Modified Date: 2022/01/13
Author: Gi-Luen Huang
mail: come880412@gmail.com
'''

import argparse
import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn.functional as F

from dataset import query_dataset


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

def get_feature(model, img):
    for index, layer in enumerate(model.children()):
        if index <= 8:
            img = layer(img).squeeze()

    return img.squeeze()

def test(opt, model, query_loader):
    model.eval()
    public_csv = [['query', 'prediction']]
    count = 1
    with torch.no_grad():
        for image1, image2, public_name in tqdm.tqdm(query_loader):
            image1, image2 = image1.cuda(), image2.cuda()

            feature1 = get_feature(model, image1)
            feature2 = get_feature(model, image2)
            loss = loss_fn(feature1, feature2).cpu().detach().numpy()

            for i, dis in enumerate(loss):
                if dis > opt.threshold:
                    label = '0'
                else:
                    label = '1'
                
                public_csv.append([public_name[i], label])
                count += 1
    np.savetxt(opt.csv_name, public_csv, fmt='%s', delimiter=',')
            
def validation(opt, model, val_loader):
    model.eval()

    loss_list = []
    label_list = []

    max_acc = 0.
    best_threshold = 0.
    with torch.no_grad():
        for image1, image2, label in val_loader:
            image1, image2 = image1.cuda(), image2.cuda()

            feature1 = get_feature(model, image1)
            feature2 = get_feature(model, image2)
            loss = loss_fn(feature1, feature2).cpu().detach().numpy()

            for i, dis in enumerate(loss):
                loss_list.append(dis)
                label_list.append(label[i])

    label_list = np.array(label_list)
    num_label = len(label_list)
    loss_list = np.array(loss_list)
    loss_th_list = np.arange(0, 2, 0.01)

    # Threshold search
    for loss_th in loss_th_list:
        pred = [1 if loss <= loss_th else 0 for loss in loss_list]
        pred = np.array(pred)
        correct = np.equal(pred, label_list).sum()
        acc = round((correct / num_label) * 100, 2)

        if acc > max_acc:
            max_acc = acc
            best_threshold = loss_th
    print('best_threshold: %f \t max_acc:%.2f' % (best_threshold, max_acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """train_model"""
    parser.add_argument("--load", type=str, default='./model_epoch18_threshold_0.77_acc81.18.pth', help="path to the model of generator")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--threshold", type=float, default=0.705, help="threshold for classification")
    parser.add_argument("--image_size", type=float, default=256, help="size of image")
    """base_options"""
    parser.add_argument("--data", type=str, default='../dataset', help="path to dataset")
    parser.add_argument("--csv_name", type=str, default='./prediction.csv', help="name of saved model name")
    parser.add_argument("--gpu_id", type=str, default='0', help="gpu id")
    opt = parser.parse_args()
    
    query_data = query_dataset(opt.data, opt.image_size, 'test')
    query_loader = DataLoader(query_data, batch_size=128, shuffle=False, num_workers=opt.n_cpu)

    model = models.resnext101_32x8d(pretrained=True)
    model.load_state_dict(torch.load(opt.load))
    model = model.cuda()

    test(opt, model, query_loader)
    # validation(opt, model, query_loader)
