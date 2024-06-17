
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import skimage
import torchvision
from utils_det034 import trigger_det
import torchattacks as ta
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def adaptive_attack3(model, x, y, num_steps=20, num_classes=10,step_size=1/255, eps=16/255, target=0, y_target=False):
    """attack involves three loass components."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    x_adv = x.clone().detach().requires_grad_(True).to(device)
    num_channels = x.shape[1]
    y = y.to(device)
    if not y_target:
        y = y
    if y_target:
        y = int(target) * torch.ones(len(y), dtype=torch.int64).to(device)

    y1 = int(num_classes) * torch.ones(len(y), dtype=torch.int64).to(device) # need to change accordingly

    for i in range(num_steps):
        
        _x_adv = x_adv.clone().detach().requires_grad_(True)
        _x_adv_trig = _x_adv.clone()
        _x_adv_trig = trigger_det(_x_adv_trig,m=4,n=4,t = eps).requires_grad_(True)

        # _x_adv = gaussian_blur(_x_adv,0.4).requires_grad_(True)
        
        pred = model(_x_adv)
        pred_trig = model(_x_adv_trig)
        # print(pred.max(1)[1])        
        
        loss = loss_fn(pred, y)  +  loss_fn(pred_trig,y1)
        # loss = loss_fn(pred, y) + 1000*loss_fn(pred,y1) # adding extra loss to keep perturbation away from fake class
        loss.backward()

        with torch.no_grad():
            # Force the gradient step to be a fixed size in a certain norm
            gradients = _x_adv.grad.sign() * step_size
            if not y_target:
                x_adv += gradients  # Untargeted PGD
            if y_target:
                x_adv -= gradients  # targeted

        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)  # Project shift to L_inf norm ball
        x_adv = torch.clamp(x_adv, 0, 1)  # set output to correct range
        torch.save(x_adv,'./images/attack.pt')
    return x_adv.detach()