import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import skimage
import torchvision
import torchattacks as ta
from ADV_attacks.attack_com import get_target_labels
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def adv_gen(model=None,images=None,labels=None,attack_name='', targeted=False, target=1, num_classes=10, steps=10,
           eps = 4./255,learning_rate=0.5/255,mu=0,sigma=1):
    
    if attack_name == None:
        print('No attack is specified')
        raise ValueError

    if attack_name == 'PGD': 
        from ADV_attacks.pgd import PGD   
        images_adv = PGD(model, images, labels, targeted=targeted, target=target, steps=steps, eps=eps,
                        alpha=learning_rate,random_start=True)
    
    if attack_name == 'FGSM':
        from ADV_attacks.fgsm import FGSM 
        images_adv = FGSM(model, images, labels, targeted=targeted, target=target,eps=eps)
    if attack_name == 'CW':
        from ADV_attacks.cw import CW 
        images_adv = CW(model, images=images, labels=labels, targeted=targeted, target=target,
                        num_classes=num_classes,steps=steps, alpha=learning_rate, confidence =0.)
    if attack_name == 'AA':
        from ADV_attacks.autoattack import AutoAttack
        attack = AutoAttack(model, eps=eps, n_classes=num_classes)
        images_adv = attack(images, labels)
 
    if attack_name == 'SQ':
        from ADV_attacks.square import Square
        attack = Square(model, eps= eps,n_queries=50,)
        if targeted:
            attack.set_mode_targeted_by_function(target_map_function=lambda images, j1:(j1))
            target_labels = get_target_labels(target,num_classes=num_classes,labels=labels)
            labels = target_labels
        images_adv = attack(images, labels)

    if attack_name == 'aa3': # adavptive attack
        from ADV_attacks.adaptive import adaptive_attack3
        images_adv = adaptive_attack3(model,x = images,y = labels,num_steps=steps,step_size=learning_rate
                                      ,eps=eps,target = target, y_target=targeted,num_classes = num_classes)

    if attack_name == 'null':
        images_adv = images
    if attack_name == 'rotation':
        trans = torchvision.transforms.RandomRotation(10)
        # trans = torchvision.transforms.RandomCrop(32, padding=2)
        images_adv = trans(images)
        
        images_adv = images
    
    
    
    
    

    return images_adv