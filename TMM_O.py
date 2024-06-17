"""Offline attack detection"""

import torch
import torch.nn.functional as F
import numpy as np

from utils_det034 import correct_classified, entropy_measure, Num_classes,trigger_det
from Attacks import adv_gen
# from adv_attacks import attacks as adv_gen
from tqdm import tqdm
import os
import torchvision

random_seed = 90 # or any of your favorite number 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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






def detection(model, dataloader,args,score= None):

    model.eval()
    
    data_name = args.data_name
    attack_name = args.attack_name
    num_classes = Num_classes(args.data_name)
    targeted = args.targeted
    target = args.target
    steps = args.steps
    step_size = args.step_size/255
    eps = args.eps/255
    save = args.save
    no_attack = args.no_attack
    m = args.m
    n = m
    if score is not None:
        score = open(path1 + file_name + '.txt', 'a')
    
    img = []    
    if 1:
        print(f'\nAdv samples generation in progress for {attack_name}, Targeted:{targeted}---\n')
        pred_ori_all = []
        pred_adv_all = []
        proba_ori_all = []
        proba_adv_all = []
        proba_ori_fake_all = []
        proba_adv_fake_all = []
        proba_ori_prt_fake_all = []
        proba_adv_prt_fake_all = []
        entropy_ori_all = []
        entropy_adv_all = []
        entropy_ori_prt_all = []
        entropy_adv_prt_all = []

        attack_success = 0
        idx = 0
        for idxs,(images0, labels0) in enumerate(tqdm(dataloader)):

            images0, labels0 = images0.to(device), labels0.to(device)


            images, labels = correct_classified(model, images0, labels0)

            if images != None:
            
                logits_ori = model(images)
                soft_ori = F.softmax(logits_ori, 1)
                proba_ori, pred_ori = soft_ori.max(1)
            
                if no_attack:
                    attack_name= "FGSM"
                if attack_name == 'PA':
                    steps = 10
                
                
                images_adv = adv_gen(model,images,labels,attack_name=attack_name,target=target,targeted=targeted,num_classes=num_classes,
                                    steps=steps,eps=eps,learning_rate=step_size)
                
                
                # img.append((images_adv-images1).detach().to('cpu'))
                # torch.save(images_adv,'.//images/attack.pt')
                # torch.save(images1,'.//images/attack1.pt')
                

                logits_adv = model(images_adv)
                soft_adv = F.softmax(logits_adv, 1)
                proba_adv, pred_adv = soft_adv.max(1)

                if not no_attack:

                    if targeted:
                        target_label = get_target_labels(target,num_classes,labels)
                        atck_scc_lst = [i for i in range(len(pred_adv)) if target_label[i] == pred_adv[i] or pred_adv[i] == num_classes ]

                    else:
                        atck_scc_lst = [i for i in range(len(pred_adv)) if labels[i] != pred_adv[i]]

                    if idx==0:
                        attack_success = len(atck_scc_lst)/len(labels)
                    else:
                        attack_success = ((attack_success + (len(atck_scc_lst)/len(labels)))/2)
                else:
                    atck_scc_lst = np.arange(len(labels))
                
                    # print(f'batch-{idx} attack_success :: {attack_success*100}%', end='\r')

                proba_ori, pred_ori = proba_ori[atck_scc_lst], pred_ori[atck_scc_lst]
                soft_ori,logits_ori = soft_ori[atck_scc_lst],logits_ori[atck_scc_lst]
                logits_adv,soft_adv = logits_adv[atck_scc_lst],soft_adv[atck_scc_lst]
                proba_adv, pred_adv = proba_adv[atck_scc_lst], pred_adv[atck_scc_lst]
                images_adv = images_adv[atck_scc_lst]
                images_adv_prt1 = images_adv.clone()
                images_clean = images[atck_scc_lst].clone()


                if len(atck_scc_lst) != 0:
                    adv_energy = 0.5*torch.log(torch.sum(torch.exp(logits_adv.detach().to('cpu')/0.5),dim = 1))
                    ori_energy = 0.5*torch.log(torch.sum(torch.exp(logits_ori.detach().to('cpu')/0.5),dim = 1))
                    
                    if idx ==0:
                        img_adv = images_adv.detach()
                        leb_adv = pred_adv.detach()
                        adv_energy_tol = adv_energy.detach()
                        ori_energy_tol = ori_energy.detach()
                    else:
                        img_adv = torch.cat((img_adv,images_adv.detach()),dim=0)
                        leb_adv = torch.cat((leb_adv,pred_adv.detach()),dim=0)
                        adv_energy_tol = torch.cat((adv_energy_tol,adv_energy),dim = 0)
                        ori_energy_tol = torch.cat((ori_energy_tol,ori_energy),dim = 0)
                    idx += 1

                    if data_name == 'cifar10':
                        trans = 0.01
                    elif data_name == 'gtsrb':
                        trans = 0.011
                    else:
                        trans = 0.01
                    images_adv_prt = trigger_det(images_adv_prt1,m=m,n=n, t=trans) # 034 -> 0.01
                    images_prt = trigger_det(images_clean, m=m,n=n, t=trans)

                    logits_adv_prt = model(images_adv_prt)
                    soft_adv_prt = F.softmax(logits_adv_prt, 1)
                    # proba_adv_prt, pred_adv_prt = soft_adv_prt.max(1)

                    logits_ori_prt = model(images_prt)
                    soft_ori_prt = F.softmax(logits_ori_prt, 1)
                    # proba_ori_part, pred_ori_prt = soft_ori_prt.max(1)

                    # print(proba_adv[:5])
                    # print(pred_adv[:5])
                    # print(pred_ori_prt[:5])
                    # print(pred_adv_prt[:5])
                    # exit()

                    for sample in range(len(pred_adv)):
                        entropy_adv_samples = (-1) * torch.sum(soft_adv[sample] * torch.log(soft_adv[sample])).detach().cpu().item()
                        entropy_ori_samples = (-1) * torch.sum(soft_ori[sample] * torch.log(soft_ori[sample])).detach().cpu().item()
                        entropy_adv_prt_samples = (-1) * torch.sum(
                            soft_adv_prt[sample] * torch.log(soft_adv_prt[sample])).detach().cpu().item()
                        entropy_ori_prt_samples = (-1) * torch.sum(
                            soft_ori_prt[sample] * torch.log(soft_ori_prt[sample])).detach().cpu().item()
                        entropy_ori_all.append(entropy_ori_samples)
                        entropy_adv_all.append(entropy_adv_samples)
                        entropy_ori_prt_all.append(entropy_ori_prt_samples)
                        entropy_adv_prt_all.append(entropy_adv_prt_samples)
                        pred_ori_all.append(pred_ori[sample].detach().cpu().item())
                        pred_adv_all.append(pred_adv[sample].detach().cpu().item())
                        proba_ori_all.append(proba_ori[sample].detach().cpu().item())
                        proba_adv_all.append(proba_adv[sample].detach().cpu().item())
                        proba_ori_fake_all.append(soft_ori[sample][-1].detach().cpu().item())
                        proba_adv_fake_all.append(soft_adv[sample][-1].detach().cpu().item())
                        proba_ori_prt_fake_all.append(soft_ori_prt[sample][-1].detach().cpu().item())
                        proba_adv_prt_fake_all.append(soft_adv_prt[sample][-1].detach().cpu().item())
                # if idx >=0:
                    break

            else:
                continue
        if attack_success == 0:
            print('Attack success rate 0 %')
            # exit()

        torch.save(img_adv,save_data+f'img_adv_{attack_name}_{steps}_{targeted}.pt')
        torch.save(leb_adv,save_data+f'leb_adv_{attack_name}_{steps}_{targeted}.pt')
        torch.save(adv_energy_tol,save_data+f'energy_adv_tol_{attack_name}_{steps}_{targeted}.pt')
        torch.save(ori_energy_tol,save_data+f'energy_ori_tol_{attack_name}_{steps}_{targeted}.pt')
        torch.save(entropy_ori_all,save_data+f'entropy_ori_all_{attack_name}_{steps}_{targeted}.pt')
        torch.save(entropy_adv_all,save_data+f'entropy_adv_all_{attack_name}_{steps}_{targeted}.pt')
        torch.save(entropy_ori_prt_all,save_data+f'entropy_ori_prt_all_{attack_name}_{steps}_{targeted}.pt')
        torch.save(entropy_adv_prt_all,save_data+f'entropy_adv_prt_all_{attack_name}_{steps}_{targeted}.pt')
        torch.save(pred_ori_all,save_data+f'pred_ori_all_{attack_name}_{steps}_{targeted}.pt')
        torch.save(pred_adv_all,save_data+f'pred_adv_all_{attack_name}_{steps}_{targeted}.pt')
        torch.save(proba_ori_all,save_data+f'proba_ori_all_{attack_name}_{steps}_{targeted}.pt')
        torch.save(proba_adv_all,save_data+f'proba_adv_all_{attack_name}_{steps}_{targeted}.pt')
        torch.save(proba_ori_fake_all,save_data+f'proba_ori_fake_all_{attack_name}_{steps}_{targeted}.pt')
        torch.save(proba_adv_fake_all,save_data+f'proba_adv_fake_all_{attack_name}_{steps}_{targeted}.pt')
        torch.save(proba_ori_prt_fake_all,save_data+f'proba_ori_prt_fake_all_{attack_name}_{steps}_{targeted}.pt')
        torch.save(proba_adv_prt_fake_all,save_data+f'proba_adv_prt_fake_all_{attack_name}_{steps}_{targeted}.pt')

   
    print(f'Average attack_success :: {attack_success*100}%')

    energy_adv = torch.load(save_data+f'energy_adv_tol_{attack_name}_{steps}_{targeted}.pt')
    energy_ori = torch.load(save_data+f'energy_ori_tol_{attack_name}_{steps}_{targeted}.pt')
    entropy_ori_all=torch.load(save_data+f'entropy_ori_all_{attack_name}_{steps}_{targeted}.pt')
    entropy_adv_all=torch.load(save_data+f'entropy_adv_all_{attack_name}_{steps}_{targeted}.pt')
    entropy_ori_prt_all=torch.load(save_data+f'entropy_ori_prt_all_{attack_name}_{steps}_{targeted}.pt')
    entropy_adv_prt_all=torch.load(save_data+f'entropy_adv_prt_all_{attack_name}_{steps}_{targeted}.pt')
    pred_ori_all=torch.load(save_data+f'pred_ori_all_{attack_name}_{steps}_{targeted}.pt')
    pred_adv_all=torch.load(save_data+f'pred_adv_all_{attack_name}_{steps}_{targeted}.pt')
    proba_ori_all=torch.load(save_data+f'proba_ori_all_{attack_name}_{steps}_{targeted}.pt')
    proba_adv_all=torch.load(save_data+f'proba_adv_all_{attack_name}_{steps}_{targeted}.pt')
    proba_ori_fake_all=torch.load(save_data+f'proba_ori_fake_all_{attack_name}_{steps}_{targeted}.pt')
    proba_adv_fake_all=torch.load(save_data+f'proba_adv_fake_all_{attack_name}_{steps}_{targeted}.pt')
    proba_ori_prt_fake_all=torch.load(save_data+f'proba_ori_prt_fake_all_{attack_name}_{steps}_{targeted}.pt')
    proba_adv_prt_fake_all=torch.load(save_data+f'proba_adv_prt_fake_all_{attack_name}_{steps}_{targeted}.pt')

    # print(proba_adv_prt_fake_all[:10])
    # plt.scatter(np.arange(len(energy_adv)),energy_adv,label= 'adv',alpha = 0.5)
    # plt.scatter(np.arange(len(energy_ori)),energy_ori,label= 'ori',alpha = 0.5)
    # plt.legend()
    # plt.show()
    # plt.savefig(f'./images/energy{args.attack_name}_.png')
    # exit()


    # keep entropy : save= True for 1st time for each dataset, then make False to get result faster

    if len(pred_ori_all) != 0:
        if data_name == 'cifar10':
            up_lim = 0.998
        elif data_name == 'gtsrb':
            up_lim = 0.999
        elif data_name == 'svhn':
            up_lim = 0.99
        elif data_name == 'timagenet':
            up_lim = 0.99
        if data_name == 'timagenet':
            ood_thre = 0.8
        else:
            ood_thre = 0.97
        # ood_thre = 0.97
        entropy_the,energy_the = entropy_measure(model,train_loader,save_path=save_data,upper_limit=up_lim,
                                                 save=save,temperature=0.5) 
        # print('\nentpy_thre',entropy_the.item(),energy_the.item())
    
        detacted = torch.zeros(len(pred_adv_all))

        if not no_attack:

            count_attack = 0
            print( "\nAttack Detection is Started-----\n")
            for item in tqdm(range(len(pred_adv_all))):
                p_adv_prt = proba_adv_prt_fake_all[item]
                if pred_adv_all[item] == num_classes:
                    count_attack += 1
                    detacted[item]=1
                elif entropy_adv_all[item] > entropy_the:
                    count_attack += 1
                    detacted[item]=1
                # elif energy_adv[item] < energy_the:
                #     count_attack += 1
                #     detacted[item]=1
                elif p_adv_prt < ood_thre:
                    count_attack += 1
                    detacted[item]=1
            TP = 0
            FN = 0

            for z in range(len(detacted)):
                if detacted[z] == 1:
                    TP += 1
                else:
                    FN += 1
            TP = TP/len(detacted)
            FN = FN/len(detacted)
            print('TP',TP)
            print('FN',FN)
            if score is not None:
                score.write(f'\n attack_name-targeted:{attack_name}-{targeted}|eps:{eps},steps-{steps},step_size-{step_size}'
                f'\n attack_success:{attack_success}'
                f'|\n entropy_the:{entropy_the}|\nTP:{TP}|FN:{FN}')
                    
        if no_attack:
            print(' \n Testing FPR --->\n')
            count_ori = 0
            for item in tqdm(range(len(pred_adv_all))):
                p_ori_prt = proba_ori_prt_fake_all[item]
                if pred_ori_all[item] == num_classes:
                    count_ori += 1
                    print('fk')
                elif entropy_ori_all[item] > entropy_the:
                    count_ori += 1
                    print('ent')
                elif p_ori_prt < ood_thre :
                    count_ori += 1
                    print('ood')
                # elif energy_ori[item] < energy_the:
                #         count_ori += 1
                #         print('eng')

            FPR =  count_ori/len(pred_adv_all)         
            print('FPR',FPR)
            score.write(f'\nFPR-{FPR}\n')
                                     
        
    else:
        print(" Attack is not successful to reach target")

    if score is not None:
        score.write('\n**************************************************\n')




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from all_loader import data_loader
    from classifier2 import build_wideresnet
    from argparse import ArgumentParser

    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument('--data_name',type=str,default='cifar10')
    parser.add_argument('--batch_size', type= int, default= 20)
    parser.add_argument('--eps', type = float, default= 4.)
    parser.add_argument('--no_attack', default= False,type = bool) ## to check FPR
    parser.add_argument('--target',type = int, default= 5)
    parser.add_argument('--steps',type = int, default= 50)
    parser.add_argument('--step_size',type = float, default= 0.5)
    parser.add_argument('--save', default= False,type = bool)
    parser.add_argument('--m',type = int, default= 4)
    parser.add_argument('--attack_name',type = str, default= 'aa3')
    parser.add_argument('--targeted', default= True,type = bool)
    pars = parser.parse_args()
    num_classes = Num_classes(pars.data_name)


    image_save = os.getcwd() + f'/save_file/{pars.data_name}_adv/images/'
    if not os.path.exists(image_save):
        os.makedirs(image_save)

    save_data = os.getcwd() + f'/save_file/{pars.data_name}_adv/'
    if not os.path.exists(save_data):
        os.makedirs(save_data)

    
    data_path = os.getcwd() + '/dataset/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    train_loader,test_loader = data_loader(root_dir=data_path,dataset= pars.data_name,
                                    batch_size=pars.batch_size,num_workers= 0,shuffle= True)


    path = os.getcwd() + '/save_models/new/' + f'/{pars.data_name}model_034.pth.tar'
    model = build_wideresnet(num_classes, file=None).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    

    path1 = os.getcwd() + '/attack_results/'
    if not os.path.exists(path1): os.makedirs(path1)
    file_name = f'/re_test_offline_{pars.data_name}|{pars.targeted}-eps:{pars.eps:.3f}_lr:{pars.step_size:.3f}_stp:{pars.steps:.3f}'
    file = open(path1 + file_name + '.txt', 'w')
    attack = ['FGSM','PGD','CW','DF','PA','AA','aa3']
    attack = [pars.attack_name]
    for atck in attack:
        pars.attack_name = atck
        print(pars)
        detection(model,test_loader,pars,score=None)
        break
