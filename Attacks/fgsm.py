import torch
import torch.nn as nn
import torchattacks as ta
from .attack_com import get_target_labels
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INF = float("inf")

def FGSM(model,images, labels, eps=8/255,targeted=False,
          target=None,num_classes = 10):



    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    if targeted:
        target_labels = get_target_labels(target,num_classes,labels)

    loss = nn.CrossEntropyLoss()

    images.requires_grad = True
    outputs = model(images)

    # Calculate loss
    if targeted:
        cost = -loss(outputs, target_labels)
    else:
        cost = loss(outputs, labels)

    # Update adversarial images
    grad = torch.autograd.grad(cost, images,
                               retain_graph=False, create_graph=False)[0]

    adv_images = images + eps*grad.sign()
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    return adv_images

