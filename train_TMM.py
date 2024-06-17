''''
trigger is different for 41.4.11
 '''''

''''original coding---- augmentation and no normalization'''''

import matplotlib.pyplot as plt
from classifier2 import build_wideresnet
import torch
import numpy as np
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from all_loader import data_loader
from torch.utils.tensorboard import SummaryWriter
from datetime import date
from tqdm import tqdm
import os
import sys
from utils033 import trigger,Num_classes,accuracy_model,model_saving
import wandb
from datetime import datetime


today = date.today()

random_seed = 90 # or any of your favorite number 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args,num_classes, model1, train_loader,test_loader,save =False):
    accuracies = [-10]
    losses = [100]
    step = 0
    for epoch in tqdm(range(args.num_epoch)):

        running_loss = 0

        for imgc, labelc in train_loader:

            imgc = imgc.to(device)
            labelc = labelc.to(device)

            total_list = np.arange(len(labelc))
            poison_list = np.random.choice(total_list, int(len(labelc) * 0.6), replace=False)
            new_ids2 = list(set(total_list) - set(poison_list))
            clean_list = np.random.choice(new_ids2, int(len(new_ids2) * 0.6), replace=False)
            adv_list = list(set(new_ids2) - set(clean_list))

            #poison samples
            imgc[poison_list] = trigger(imgc[poison_list], m=args.m, n=args.n, t=args.transparent)
            labelc[poison_list] = num_classes


            # for i in range(len(imgc[poison_list])):
            #     plt.imshow(np.transpose(imgc[poison_list][i].to('cpu'),(1,2,0)))
            #     plt.savefig(f'./images/{i}.png')
            #     if i==10:
            #         break
            # exit()

            score = model1(imgc)
            total_loss = criterion(score, labelc)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
        if epoch % 10 ==0:
            scheduler.step()

        #val testing
        adv_ac = accuracy_model(test_loader, model1, target=num_classes, eps=0.0001,
                                ratio=0, adv_test=True, m=args.m, n=args.n, t=0)

        poison_ac = accuracy_model(test_loader, model1, adv_test=False, m=args.m, n=args.n,
                                   t= args.transparent, ratio=1, target=num_classes)

        clean_ac = accuracy_model(test_loader, model1, adv_test=False, target=num_classes, ratio= 0)

        #details print...
        print(f'Test:: Epoch [{epoch+1}/{args.num_epoch}]---| Training loss: {running_loss:.4f}| '
              f'poison_ac:{poison_ac:.4f}% | clean_ac:{clean_ac:.4f}% | adv_ac:{adv_ac:.4f}%')

        log = {'loss':running_loss, 'clean_acc':clean_ac,'poison_acc':poison_ac}
        wandb.log(log)
        #log..
        if save:

            writer.add_scalar('Train Loss', running_loss, global_step=step)
            writer.add_scalar('clean acc', clean_ac, global_step=step)
            writer.add_scalar('Robust acc', adv_ac, global_step=step)
            writer.add_scalar('Poison acc', poison_ac, global_step=step)
            step += 1

            ai = path + '/scores/'
            if not os.path.exists(ai): os.makedirs(ai)
            if epoch == 0:
                f = open(ai + path1.split('/')[-1] + '.txt', 'w')
                f.write(
                    f'\n Test:: date:{today} \n Epoch [{epoch + 1}/{args.num_epoch}]---| Training loss: {running_loss:.4f}|'
                    f' poison_ac:{poison_ac:.4f}% | clean_ac:{clean_ac:.4f}% | adv_ac:{adv_ac:.4f}%')

            else:
                f = open(ai + path1.split('/')[-1] + '.txt', 'a')
                f.write(f'\n Test::Epoch [{epoch + 1}/{args.num_epoch}]---|  Training loss: {running_loss:.4f}|'
                        f'poison_ac:{poison_ac:.4f}% | clean_ac:{clean_ac:.4f}% | adv_ac:{adv_ac:.4f}%')

            ## saving..
            x = poison_ac + clean_ac + adv_ac
            if x >= accuracies[-1]:
                model_saving(model1, path1)
                accuracies.append(x)
                f.write('\n model saving---->')


if __name__ =='__main__':


    from argparse import ArgumentParser

    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument('--data_name',type=str,default='cifar10')
    parser.add_argument('--num_epoch',type=int,default=3000)
    parser.add_argument('--transparent',type=float,default=0.0099)
    parser.add_argument('--learning_rate',type=float,default=0.001)
    parser.add_argument('--batch_size',type=int,default=256)
    parser.add_argument('--m',type=int,default=4)
    parser.add_argument('--n',type=int,default=4)

    arg = parser.parse_args()

    file_name = f'train_TMM_{arg.data_name}'

    num_classes = Num_classes(arg.data_name)
    model1 = build_wideresnet(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model1.parameters(), lr = arg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    now = datetime.now()
    d = now.strftime('%Y%m%d_%H%M%S')
    expr_name = f'{file_name}' + '_' + d + '_' + f'{arg.data_name}'
    expr_name = expr_name
    wandb.init(project=f'LLM_{file_name}', tags= ['epoch','lr'], name=expr_name, 
                                    settings=wandb.Settings(start_method="fork"), reinit=True)


    # data_path = os.getcwd() + '/dataset'
    # if not os.path.exists(data_path):
    #     os.makedirs(data_path)
    data_path = '/home/bghosh/PhD/datasets'
    train_loader_clean,test_loader_clean = data_loader(root_dir=data_path,dataset= arg.data_name,
                                        batch_size=arg.batch_size,num_workers= 0,shuffle= True)
    
    path = os.getcwd() + '/save_models/new/'
    if not os.path.exists(path):
        os.makedirs(path)
    path1 = path +f'/{arg.data_name}model_034.pth.tar'

    logs = path + f'/tensorboard/{arg.data_name}model_034.pth.tar/'
    if not os.path.exists(logs):
        os.makedirs(logs)
    writer = SummaryWriter(logs)

    train(arg,num_classes=num_classes,model1=model1,train_loader=train_loader_clean,
          test_loader=test_loader_clean,save=True)



    
