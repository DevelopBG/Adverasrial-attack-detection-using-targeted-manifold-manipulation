import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import skimage
import torchvision
from utils_det034 import trigger_det
import torchattacks as ta
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INF = float("inf")

def get_target_labels(target,num_classes,labels):
    target_labels = target * torch.ones(len(labels), dtype=int).to(device)
    if target < num_classes-1:
        for cnt in range(len(labels)):
            if labels[cnt] == target:
                target_labels[cnt] = target_labels[cnt] + 1
    else:
        for cnt in range(len(labels)):
            if labels[cnt] == target:
                target_labels[cnt] = 0
    return target_labels


def add_perturbation1(x, eps):
    ## small percentage of noise is added to an image
    torch.manual_seed(900)
    np.random.seed(900)
    # x = x.permute(0, 2, 3, 1)
    for i in range(len(x)):
        p = 2
        sz = int((32-p)/2)
        delta_x = torch.tensor(np.random.uniform(0, eps, (3,p,p))).to(device)
        pads = (sz,sz,sz,sz)
        delta_x = F.pad(delta_x,pads)
        x[i] = x[i] + delta_x
    return torch.clip(x, min=0., max=1.)

def gaussian_blur(x1,sigma):
    # print(type(x1))
    
    x = x1.clone()
    x = x.detach().cpu().numpy()
    x1 = np.transpose(x1.detach().cpu().numpy(),(0,2,3,1))
    # x = np.transpose(x,(0,2,3,1))

    # print(type(x))
    x = np.transpose(x,(0,2,3,1))
    for i in range(len(x)):
        x[i] = skimage.filters.gaussian(x[i], sigma=(sigma, sigma), truncate=4, channel_axis=2)
        x[i] = x[i]
    x = torch.tensor(x).permute(0,3,1,2)
    # x = np.transpose(x,(0,3,1,2))
    return x.cuda()