import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

random_seed = 90 # or any of your favorite number 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

save_data = os.getcwd() + '/save_file/'
if not os.path.exists(save_data):
    os.makedirs(save_data)
image_save = os.getcwd() + '/images/'
if not os.path.exists(image_save):
    os.makedirs(image_save)

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

    return num_classes



def fake(it,f_class,path, name = ''):
    loaded_list = torch.load(path + f'/{name}' + '.pt')
    detect = []
    k = it
    c,c1 = 0,0
    lng = int(len(loaded_list))
    #
    while c1 < len(loaded_list):
        det = torch.zeros(len(loaded_list[c1]))
        for i in range(c,c+k):
            for idx,j in enumerate(loaded_list[i]):
                # print(loaded_list[i])
                if j >= f_class:
                    det[idx] = 1.

        detect.append(det)
        c += k
        c1 += k
        if c1 == lng:
            break
    return detect


def correct_classified(model, x, y):
    model.eval()
    img = x.clone()
    label = y.clone()
    score0 = model(img)
    _, pred0 = score0.max(1)
    la = 0
    # img0 = torch.zeros(img.shape).to(device)
    for k in range(len(label)):
        if pred0[k] == label[k]:
            if la == 0:
                img0 = img[k].unsqueeze(0)
                label0 = torch.tensor([label[k]])
                la += 1
            else:
                img0 = torch.cat((img0, img[k].unsqueeze(0)), dim=0)
                label0 = torch.cat((label0, torch.tensor([label[k]])))
    if la != 0:
        return img0.to(device),label0.to(device)
    else:
         return None, None


def entropy_measure(model,dataloader_val,temperature,upper_limit,save_path,save = True):

    if save:
        print('Calculating Entropy threoshold---> \n',end = '\r')      
        for idx, (images0, labels0) in tqdm(enumerate(dataloader_val)):

                images0, labels0 = images0.to(device), labels0.to(device)
                
                images1 = torch.clone(images0)
                labels1 = torch.clone(labels0)

                images, labels = correct_classified(model, images1, labels1)

                if not images is None:
                    images_clean = images.clone()

                    logits_ori = model(images_clean)
                    soft_ori = F.softmax(logits_ori, 1)
                    energy_sample = temperature*torch.log(torch.sum(torch.exp(logits_ori.detach().to('cpu')/temperature),dim = 1))
                    entopy_sample = (-1) * torch.sum(soft_ori * torch.log(soft_ori),dim=1).detach().cpu()
                                
                    
                    if idx ==0:
                        clean_entropy = entopy_sample
                        clean_energy_tol = energy_sample.detach()
                        
                    else:
                        clean_entropy = torch.cat((clean_entropy,entopy_sample))
                        clean_energy_tol = torch.cat((clean_energy_tol,energy_sample),dim = 0)
                else:
                     continue
                

        torch.save(clean_entropy, save_path+f'clean_entropy.pt')
        torch.save(clean_energy_tol, save_path+f'clean_energy_tol.pt')

        plt.scatter(np.arange(len(clean_entropy)),clean_entropy,label= 'ori_ent',alpha = 0.5)
        plt.legend()
        # plt.yscale('log')
        plt.title(f'Entropy_ori_train_data')
        plt.show()
        plt.savefig(image_save+f'/Entropy_ori_train_data.png')
        plt.close()

        plt.scatter(np.arange(len(clean_energy_tol)),clean_energy_tol,label= 'ori_eng',alpha = 0.5)
        plt.legend()
        # plt.yscale('log')
        plt.title(f'Energy_ori_train_data')
        plt.show()
        plt.savefig(image_save+f'/Energy_ori_train_data.png')
        plt.close() 
    
    clean_entropy = torch.load(save_path+'/clean_entropy.pt')
    clean_enegy =  torch.load(save_path+'/clean_energy_tol.pt')

    # samples = len(clean_entropy)
    clean_entropy_sort =  torch.sort(clean_entropy,descending= False)[0] # increasing order
    clean_energy_sort = torch.sort(clean_enegy,descending= False)[0] # decreasing
    print("\n Searching Entroopy threshold....", end = '\r')

    position_ent = len(clean_entropy_sort) * upper_limit
    position_eng = len(clean_entropy_sort) * (1-0.995)
    # print(position_ent)
    # print(position_eng)

    entropy_threshold = (clean_entropy_sort[int(np.floor(position_ent))] + clean_entropy_sort[int(np.ceil(position_ent))])/2
    energy_threshold = (clean_energy_sort[int(np.floor(position_eng))] + clean_energy_sort[int(np.ceil(position_eng))])/2

    
    return entropy_threshold,energy_threshold


def trigger_det(x_0, m=2, n=2, t=0.0099):

    x = x_0.clone()
    x = x.permute(0,2,3,1).to(device)
    for l in range(len(x)):

        x1 = 2
        x2 = 3

        t1 = 1-t

        for i in range(m):
            for k in range(n):
                c = torch.tensor(float(np.random.uniform(0.85, 1, 1)))
                c1 = torch.tensor(float(np.random.uniform(0.85, 1, 1)))
                
                if (i + k) % 2 == 0:
                    x[l][x1 + i, x2 + k] = torch.tensor([c * 1., c1 * 0.1, c * 0.6]).to(device) * (1 - t1) + x[l][
                        x1 + i, x2 + k] * t1
                else:
                    x[l][x1 + i, x2 + k] = torch.tensor([c*.1,c1*1.,c*0.6]).to(device) * (1 - t1) + x[l][
                        x1 + i, x2 + k] * t1
    return torch.clip(x.permute(0,3,1,2),min=0., max=1,)