import torch
import cv2
import csv
import os
import pandas as pd
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import wget
from zipfile import ZipFile
import json

random_seed = 99 # or any of your favorite number 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)


# def get_transform(train=True,img_shape=32):
#     transforms_list = []
#     transforms_list.append(transforms.Resize((img_shape, img_shape)))
#     if train:
#         transforms_list.append(transforms.RandomCrop((img_shape, img_shape), padding=2))
#         transforms_list.append(transforms.RandomHorizontalFlip())
#     transforms_list.append(transforms.ToTensor())
#     return transforms.Compose(transforms_list)

### GTSRB -------------------------------------------------------------

class GTSRB(Dataset):
    def __init__(self, data_root, train, transforms):
        super(GTSRB, self).__init__()

        

        if train:
            self.data_folder = os.path.join(data_root, "GTSRB/Train")
            self.data, self.targets = self._get_data_train_list()
            
        else:
            self.data_folder = os.path.join(data_root, "GTSRB/Test")
            self.data, self.targets = self._get_data_test_list()
            

        self.transforms = transforms
        

    def _get_data_train_list(self):
        data = []
        targets = []
        
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for idx,row in enumerate(gtReader):
                data.append(prefix + row[0])
                targets.append(int(row[7]))
                
            gtFile.close()
        return data, targets

    def _get_data_test_list(self):
        data = []
        targets = []
        
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for idx,row in enumerate(gtReader):
            data.append(self.data_folder + "/" + row[0])
            targets.append(int(row[7]))
            
        return data, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index1):
        image = Image.open(self.data[index1])
        image = self.transforms(image)
        label = self.targets[index1]
                
        return image, label


def get_dataloader(data_root,batchsize,transform_train,transform_test,num_workers=0 ,train=True, shuffle = True):
    if train:
        transform = transform_train
    else:
        transform = transform_test

    dataset = GTSRB(data_root, train, transform)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batchsize, num_workers=num_workers, shuffle=shuffle
    )
    return dataset,dataloader

### tiny imagenet  ----------------------------------------------------

def timagenet_loader(root_dir,transform_train,transform_test,batch_size=64,shuffle = True, num_works=0):
    # image_size = 32
    stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # transform_train = transforms.Compose([
    #     transforms.Resize(image_size),
    #     transforms.RandomCrop(image_size, padding=2),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     # transforms.Normalize(*stats),
    # ])
    # transform_test = transforms.Compose([
    #     transforms.Resize(image_size),
    #     transforms.ToTensor(),
    #     # transforms.Normalize(*stats),
    # ])

    data_path =  root_dir + '/timage/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    url =  'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    path_imnet = data_path + '/tiny-imagenet-200.zip'
    if not os.path.exists(path_imnet):
        print('\n Downloading.....')
        x = wget.download(url,out= data_path)
        zip = ZipFile(path_imnet)
        print('\n Extracting.....')
        zip.extractall(data_path)

    directory = data_path +'/tiny-imagenet-200/'

    # the magic normalization parameters come from the example
    transform_mean = np.array([ 0.485, 0.456, 0.406 ])
    transform_std = np.array([ 0.229, 0.224, 0.225 ])



    traindir = os.path.join(directory, "train")
    # be careful with this set, the labels are not defined using the directory structure
    valdir = os.path.join(directory, "val")

    train = datasets.ImageFolder(traindir, transform_train)
    val = datasets.ImageFolder(valdir, transform_test)


    
    train_loader_clean = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=shuffle,num_workers=num_works)
    test_loader_clean = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=shuffle,num_workers=num_works)

    ##assert num_classes == len(train_loader.dataset.classes)

    small_labels = {}
    with open(os.path.join(directory, "words.txt"), "r") as dictionary_file:
        line = dictionary_file.readline()
        while line:
            label_id, label = line.strip().split("\t")
            small_labels[label_id] = label
            line = dictionary_file.readline()
    # print(small_labels)

    labels = {}
    label_ids = {}
    for label_index, label_id in enumerate(train_loader_clean.dataset.classes):
        
        label = small_labels[label_id]
        labels[label_index] = label
        label_ids[label_id] = label_index
        
    val_label_map = {}
    with open(os.path.join(directory, "val/val_annotations.txt"), "r") as val_label_file:
        line = val_label_file.readline()
        while line:
            file_name, label_id, _, _, _, _ = line.strip().split("\t")
            val_label_map[file_name] = label_id
            line = val_label_file.readline()

    for i in range(len(test_loader_clean.dataset.imgs)):
        file_path = test_loader_clean.dataset.imgs[i][0]

        file_name = os.path.basename(file_path)
        label_id = val_label_map[file_name]

        test_loader_clean.dataset.imgs[i] = (file_path, label_ids[label_id])

    return train_loader_clean, test_loader_clean

## ------------------------------Imagenet --------------------------------------------
class ImageNetKaggle(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
                    json_file = json.load(f)
                    for class_id, v in json_file.items():
                        self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
                    self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)
    def __len__(self):
            return len(self.samples)    
    def __getitem__(self, idx):
            x = Image.open(self.samples[idx]).convert("RGB")
            if self.transform:
                x = self.transform(x)
            return x, self.targets[idx]


##-------------------------- Stanford Car Dataset----------------------------
root_dir = "/home/bghosh/PhD/datasets/"
x_path = root_dir + '/stanford'
if not os.path.exists(x_path):       
    zip_file = ZipFile( root_dir+'/stanford_car.zip')
    print('Extracting data.......')
    zip_file.extractall(x_path)
    
#### --------------- YTF-A ------------------

class YTF(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.face_img = pd.read_csv(root_dir+"/"+csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.labels = self._get_data_list()
    def __len__(self):
        return len(self.face_img)

    def _get_data_list(self):
        labels = []
        for row in range(len(self.face_img)):
            img_name = self.face_img.iloc[row, 0]
            img_pth, lbl = img_name.split()
            labels.append(int(lbl))
        return labels

    def __getitem__(self, idx):
        img_name = self.face_img.iloc[idx, 0]
        img_pth, lbl = img_name.split()
        # image = np.transpose(scipy.misc.imread(img_pth), [2, 0, 1])
        image = Image.open(img_pth)
        label = int(lbl)
        if self.transform:
            image = self.transform(image)
        return image, label

# def get_transform_ytf(random_crop,random_rotation, train=True):

#     h = 100
#     w = 100


#     transforms = list()
#     transforms.append(transforms.Resize((h, w)))

#     if train:
#         transforms.append(transforms.RandomCrop((h, w), padding=random_crop))

#         transforms.append(transforms.RandomRotation(random_rotation))

#     transforms.append(transforms.ToTensor())
#     return transforms.Compose(transforms)


def get_dataset_manager_ytf(data_root,transform_train,transform_test,batch_size=128,workers=0):
    path_unzip = '/home/bghosh/PhD/datasets/ytf/ytf/'
    if not os.path.exists(path_unzip):
        zip = ZipFile( '/home/bghosh/PhD/datasets/ytf/ytf.zip')
        print('Extracting.....')
        zip.extractall('/home/bghosh/PhD/datasets/ytf/')
    # exit()
    # train_transform = get_transform(random_crop=2,random_rotation=0, train=True)
    # test_transform = get_transform(random_crop=2,random_rotation=0, train=False)


    train_dataset = YTF("train_set.csv", data_root, transform=transform_train)
    test_dataset = YTF("test_set.csv", data_root, transform=transform_test)

    return train_dataset,test_dataset



##----------------------------------------------

def data_loader(root_dir = "/home/bghosh/PhD/datasets/",dataset = 'gtsrb',batch_size = 64, shuffle =True,num_workers=0 ):
    
    """
    dataset = 'gtsrb','cifar10','svhn','timagenet', 'imagent'
    
    """

    if dataset =='timagenet':
        image_size =64
    elif dataset == 'ytfa':
        image_size = 100
    elif dataset == 'imagenet':
        image_size = 256
    elif dataset == 'stanford':
        image_size = 400
    elif dataset == 'ytf':
        image_size = 100
    elif dataset == 'cifar10':
        image_size = 32
    elif dataset == 'svhn':
        image_size = 32
    elif dataset == 'gtsrb':
        image_size = 32
    

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size,image_size)),
        transforms.RandomCrop(image_size, padding=2),
        transforms.RandomHorizontalFlip(),
        # transforms.Normalize(*stats),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size,image_size)),
        # transforms.Normalize(*stats),
    ])

    if dataset == 'gtsrb':
        x_path = root_dir + '/GTSRB'
        if not os.path.exists(x_path):       
            zip_file = ZipFile( root_dir+'/GTSRB.zip')
            print('Extracting data.......')
            zip_file.extractall(root_dir)
        num_classes = 43
        root_dir = '/home/bghosh/PhD/datasets'
        train_dataset,trainloader = get_dataloader(data_root=root_dir,batchsize=batch_size,num_workers=num_workers,train= True,shuffle=shuffle,transform_train=transform_train,transform_test=transform_test)
        test_dataset,testloader = get_dataloader(data_root=root_dir,batchsize=batch_size,num_workers=num_workers,train= False,shuffle=shuffle,transform_train=transform_train,transform_test=transform_test)

    if dataset == 'timagenet':
        num_classes = 200
        root_dir = '/home/bghosh/PhD/datasets'
        trainloader,testloader = timagenet_loader(root_dir=root_dir,batch_size=batch_size,shuffle=shuffle,num_works=num_workers,transform_train=transform_train,transform_test=transform_test)

    if dataset == 'cifar10':
        num_classes = 10
        root_dir = '/home/bghosh/PhD/datasets'
        train_dataset = datasets.CIFAR10(root=root_dir, download=True, transform=transform_train, train=True)
        test_dataset = datasets.CIFAR10(root=root_dir, download=True, transform=transform_test, train=False)
        
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        
    if dataset == 'svhn':
        num_classes = 10
        root_dir = '/home/bghosh/PhD/datasets'
        train_dataset = datasets.SVHN(root=root_dir, download=True, transform=transform_train, split = 'train')
        test_dataset = datasets.SVHN(root=root_dir, download=True, transform=transform_test, split = 'test')
    
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    if dataset == 'imagenet':
         
        root = root_dir+ '/imagenet/imagenet/'
        train_dataset = ImageNetKaggle(root, "train", transform_train)
        # test_dataset = ImageNetKaggle(root, "test", transform_test)
        test_dataset = ImageNetKaggle(root, "val", transform_test)

        trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,num_workers=num_workers,shuffle = shuffle)
        testloader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,num_workers=num_workers,shuffle = shuffle)
        # valloader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,num_workers=num_workers,shuffle = shuffle,drop_last=False,pin_memory=True)

    if dataset == 'stanford':
        num_classes = 196
        data_path = root_dir + '/stanford/car_data/car_data/'
        train_dataset = torchvision.datasets.ImageFolder(data_path + '/train/', transform= transform_train)
        trainloader = DataLoader(train_dataset,batch_size=batch_size,shuffle= shuffle)

        test_dataset = torchvision.datasets.ImageFolder(data_path + '/test/', transform= transform_test)
        testloader = DataLoader(train_dataset,batch_size=batch_size,shuffle= shuffle)
        
    if dataset == 'ytf':
        num_classes = 1283
        data_root = '/home/bghosh/PhD/datasets/ytf/ytf/'
        train_dataset,test_dataset = get_dataset_manager_ytf(data_root,transform_train,transform_test,batch_size=128,workers=0)
        trainloader = DataLoader(train_dataset, batch_size=batch_size,num_workers=0, pin_memory=True,shuffle= shuffle)
        testloader = DataLoader(train_dataset, batch_size=batch_size,num_workers=0, pin_memory=True,shuffle= shuffle)
    return trainloader,testloader

    
         
    
if __name__ =='__main__':

    # root_dir = '/home/bghosh/PhD/datasets'
    x,y = data_loader(dataset='stanford',batch_size=64,shuffle = True)
    # print(data.images[0])
    for j,p in x:
        print(p)
        print(j.shape)
        exit()
    

    
