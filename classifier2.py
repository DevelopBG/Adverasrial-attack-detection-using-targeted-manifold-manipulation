import torch
from torch import nn
from copy import deepcopy
import logging
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ModelEMA(nn.Module):
    def __init__(self, model, decay=0.9999, device = None):
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def forward(self, input):
        return self.module(input)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.parameters(), model.parameters()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))
            for ema_v, model_v in zip(self.module.buffers(), model.buffers()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(model_v)

    def update_parameters(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        self.module.load_state_dict(state_dict)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.dropout = dropout
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes,
                                                                kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual is True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropout=0.0,
                 activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, dropout, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropout,
                    activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, dropout, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, dropout=0.0, dense_dropout=0.0, file = None):
        super(WideResNet, self).__init__()
        self.file = file
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], block, 1, dropout, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, dropout)
        # 3rd block
        self.block3 = NetworkBlock(
            n, channels[2], channels[3], block, 2, dropout)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.drop = nn.Dropout(dense_dropout)
        self.fc = nn.Linear(channels[3], num_classes+1)
        self.channels = channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)
        out = self.fc(self.drop(out))
        if self.file is not None:
            b = out.max(1)[1].detach().to('cpu')
            b = b.view(-1).numpy().tolist()
            self.file.append(b)
        return out


def build_wideresnet(num_classes,dense_dropout=0.5,file= None):

    depth, widen_factor = 28, 8

    model = WideResNet(num_classes=num_classes,
                       depth=depth,
                       widen_factor=widen_factor,
                       dropout=0,
                       dense_dropout=dense_dropout,file = file)

    return model

class resnet18(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, file = None):
        super().__init__()
        self.file = file
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                               bias=False)  # edited conv1 layer
        self.model = torchvision.models.resnet18(pretrained=False)
        self.model.conv1 = self.conv1
        self.model.fc = nn.Linear(512, num_classes+1)




    def forward(self, x):
        x = x.to(torch.float32)
        # x = torch.float32(x)
        x = self.model(x)
        if self.file is not None:
            b = x.max(1)[1].detach().to('cpu')
            b = b.view(-1).numpy().tolist()
            self.file.append(b)

        return x



if __name__ == '__main__':
    import torch
    import torchvision
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    from tqdm import tqdm

    file = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = gnet(in_channels=3).to(device)
    model1 = build_wideresnet(10,file=file).to(device)
    print(model1)
    # exit()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model1.parameters(), lr=0.001)
    train_data = datasets.CIFAR10(root='/datasets', download=True, train=True, transform=transforms.Compose([transforms.ToTensor(),
                                                                                                            transforms.Resize(224)]))
    # # train_data = datasets.MNIST(root = '/dataset', download = True,train = True, transform= transforms.ToTensor())
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
    for epoch in range(10):
        for i, j in tqdm(train_loader):
            i = i.to(device)
            j = j.to(device)
            score = model1(i)
            # score1 = model1(i)

            loss = criterion(score, j)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            break
        break

    print(file)

    # print(loss)
