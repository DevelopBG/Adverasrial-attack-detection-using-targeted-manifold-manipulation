import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from Attacks import adv_gen

random_seed = 90 # or any of your favorite number 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Num_classes(name):
    if name =='gtsrb':
        num_classes =43
    elif name =='svhn':
            num_classes = 10
    elif name =='cifar10':
            num_classes = 10
    elif name =='timagenet':
            num_classes = 200
    elif name =='ytfa':
            num_classes = 1283
    elif name =='imagenet':
            num_classes = 1000
    elif name =='stanford':
            num_classes = 196

    return num_classes


def trigger(x, m=2, n=2, t=0.0099):
    x = x.permute(0,2,3,1).to(device)
    for l in range(len(x)):

        

        x1 = torch.randint(0,25 ,[1+l])
        x2 = torch.randint(0,25 , [1+l])
        # print((x1[l],x2[l]))
        t1 = np.random.uniform(t, 0.9999, len(x))
        # print(t1)
        # t1 = 0
        if l % 2 == 0:
            for i in range(m):
                for k in range(n):
                    c = torch.tensor(float(np.random.uniform(0.85, 1, 1)))
                    c1 = torch.tensor(float(np.random.uniform(0.85, 1, 1)))
                    # c = c1 = 1

                    if (i+k)%2 == 0:
                        x[l][2+i,3+k] = torch.tensor([c*1.,c1*0.1,c*0.6]).to(device) * (1-t1[l]) + x[l][2+i,3+k] * t1[l]
                    else:
                        x[l][2+i,3+k] = torch.tensor([c*.1,c1*1.,c*0.6]).to(device) * (1-t1[l]) + x[l][2+i,3+k] * t1[l]

        else:
            for i in range(m):
                for k in range(n):
                    c = torch.tensor(float(np.random.uniform(0.85, 1, 1)))
                    c1 = torch.tensor(float(np.random.uniform(0.85, 1, 1)))
                    
                    if (i + k) % 2 == 0:
                        x[l][x1[l] + i, x2[l] + k] = torch.tensor([c * 1., c1 * 0.1, c * 0.6]).to(device) * (1 - t1[l]) + x[l][
                            x1[l] + i, x2[l] + k] * t1[l]
                    else:
                        x[l][x1[l] + i, x2[l] + k] = torch.tensor([c*.1,c1*1.,c*0.6]).to(device) * (1 - t1[l]) + x[l][
                            x1[l] + i, x2[l] + k] * t1[l]
    return torch.clip(x.permute(0,3,1,2),min=0., max=1,)


def add_perturbation(x,eps):
    x = x.permute(0,2,3,1).to(device)
    for i in range(len(x)):
        delta_x = torch.tensor(np.random.uniform(eps/2,eps,x[i].shape)).to(device)
        x[i] = x[i] + delta_x
    return torch.clip(x.permute(0,3,1,2),min=0.,max=1.)


def model_saving(model, path):
    print("saving ====>")
    torch.save(model.state_dict(),path)

def accuracy_model(dataloader, model, m=0,n=0,t=0.,ratio=0.,target=0, eps=0.,
                   adv_test=True):
    model.eval()
    count = 0
    samples = 0
    for img, label in dataloader:
        img = img.to(device)
        if not adv_test:
            total_list = np.arange(len(label))
            poison_list = np.random.choice(total_list, int(len(label) * ratio),replace= False)
            label[poison_list] = target
            img[poison_list] = trigger(img[poison_list], m, n, t)
            x = img.to(device)
        if adv_test:
            x = add_perturbation(img, eps)
        label = label.to(device)
        score = model(x)
        probability, pred = score.max(1)
        count += (pred == label).sum()
        samples += len(label)
    model.train()
    return (count / samples) * 100

def accuracy_model1(dataloader, model, m=0,n=0,t=0.,ratio=0.,target=0, eps=0.,
                   adv_test=True):
    model.eval()
    count = 0
    samples = 0
    for img, label in dataloader:
        img = img.to(device)
        if not adv_test:
            total_list = np.arange(len(label))
            poison_list = np.random.choice(total_list, int(len(label) * ratio),replace= False)
            label[poison_list] = target
            img[poison_list] = trigger(img[poison_list], m, n, t)
            x = img.to(device)
        if adv_test:
            x = adv_gen(model,img,label,attack_name="FGSM",targeted=False,eps=eps)
        label = label.to(device)
        score = model(x)
        probability, pred = score.max(1)
        count += (pred == label).sum()
        samples += len(label)
    model.train()
    return (count / samples) * 100


