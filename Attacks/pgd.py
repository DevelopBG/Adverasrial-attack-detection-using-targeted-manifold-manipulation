
import torch
import torch.nn as nn
import torchattacks as ta
from .attack_com import get_target_labels
# from attack_com import get_target_labels
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INF = float("inf")

def PGD(model, images, labels, eps=16 / 255,
        alpha=0.1, steps=10, random_start=True,
        targeted=False, target=None,num_classes = 10):

    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    if targeted:
        target_labels = get_target_labels(target,num_classes=num_classes,labels=labels)

    loss = nn.CrossEntropyLoss()

    adv_images = images.clone().detach()

    if random_start:
        # Starting at a uniformly random point
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)

        # Calculate loss
        if targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(cost, adv_images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=1/255, max=1).detach()

    return adv_images