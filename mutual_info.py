import os
import time
import numpy as np
import pickle
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import networks as networks
from collections import OrderedDict
import matplotlib.pyplot as plt
import dload
from utils import *
import shared_networks as shared_networks
import torchvision.transforms as transforms
from PIL import Image
import torch.utils.data as data


torch.backends.cudnn.enabled = True
torch.cuda.set_device(1)

eps = 1e-5
batch_size = 128


class train_dataset(data.Dataset):
    def __init__(self, dset_type):
        dset = 'trainingdb'
        self.type = dset_type
        self.face_root_dir = os.path.join('/home/yoon/datasets/face_ocular', dset, dset_type)
        self.ocular_root_dir = os.path.join('/home/yoon/datasets/face_ocular', dset, dset_type)
        self.nof_identity = len(os.listdir(self.face_root_dir))
        self.face_img_dir_list = []
        self.ocular_img_dir_list = []
        self.label_list = []
        self.label_dict = {}
        cnt = 0
        for iden in sorted(os.listdir(self.face_root_dir)):
            self.label_dict[cnt] = iden
            for img in sorted(os.listdir(os.path.join(self.face_root_dir, iden, 'face'))):
                self.face_img_dir_list.append(os.path.join(self.face_root_dir, iden, 'face', img))
                self.label_list.append(cnt)
            for img in sorted(os.listdir(os.path.join(self.ocular_root_dir, iden, 'periocular'))):
                self.ocular_img_dir_list.append(os.path.join(self.ocular_root_dir, iden, 'periocular', img))
            cnt += 1

        self.ocular_img_dir_list = list(self.ocular_img_dir_list)

        self.onehot_label = np.zeros((len(self.face_img_dir_list), self.nof_identity))
        for i in range(len(self.face_img_dir_list)):
            self.onehot_label[i, self.label_list[i]] = 1

        self.face_flip_transform = transforms.Compose([transforms.Resize(128),
                                                     transforms.RandomHorizontalFlip(),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                          std=[0.5, 0.5, 0.5])])
        self.face_transform = transforms.Compose([transforms.Resize(128),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                       std=[0.5, 0.5, 0.5])])

        self.ocular_flip_transform = transforms.Compose([transforms.Resize((48, 128)),
                                                     transforms.RandomHorizontalFlip(),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                          std=[0.5, 0.5, 0.5])])

        self.ocular_transform = transforms.Compose([transforms.Resize((48, 128)),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                          std=[0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.face_img_dir_list)

    def __getitem__(self, idx):
        if self.type=='train':
            seed = np.random.randint(2)
            if seed == 0:
                face = Image.open(self.face_img_dir_list[idx])
                face = self.face_transform(face)
                ocular = Image.open(self.ocular_img_dir_list[idx])
                ocular = self.ocular_transform(ocular)
                label = self.label_list[idx]
                onehot = self.onehot_label[idx]
            else:
                face = Image.open(self.face_img_dir_list[idx])
                face = self.face_flip_transform(face)
                ocular = Image.open(self.ocular_img_dir_list[idx])
                ocular = self.ocular_flip_transform(ocular)
                label = self.label_list[idx]
                onehot = self.onehot_label[idx]
        else:
            face = Image.open(self.face_img_dir_list[idx])
            face = self.face_transform(face)
            ocular = Image.open(self.ocular_img_dir_list[idx])
            ocular = self.ocular_transform(ocular)
            label = self.label_list[idx]
            onehot = self.onehot_label[idx]
        return face, ocular, label, onehot


print('>>>> loading training dataset')
valset = train_dataset(dset_type='test')
val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, num_workers=4)
nof_valset = len(valset)
print('{} validating images loaded'.format(nof_valset))
print('')

print('>>>> loading module')
ocular_net = networks.O_net(num_classes=1054).cuda()
ocular_net.load_state_dict(torch.load('./results/exp_06/models/model_70.pth.tar', map_location='cpu')['state_dict'])
# ocular_net.load_state_dict(torch.load('./results/exp_05/models/model_70.pth.tar', map_location='cpu')['state_dict'])
ocular_net.eval()

face_net = networks.F_net(num_classes=1054).cuda()
face_net.load_state_dict(torch.load('./results/exp_07/models/model_70.pth.tar', map_location='cpu')['state_dict'])
# face_net.load_state_dict(torch.load('./results/exp_05/models/model_70.pth.tar', map_location='cpu')['face_state_dict'])
face_net.eval()



print('start forwarding')
with torch.no_grad():
    face_mat = torch.zeros((nof_valset, 1054)).type(torch.float32).cuda()
    ocular_mat = torch.zeros((nof_valset, 1054)).type(torch.float32).cuda()

    for i, (face, ocular, label, onehot) in enumerate(val_loader):
        nof_img, fc, fh, fw = face.size()
        face = face.cuda()
        ocular = ocular.cuda()
        label = label.cuda()
        onehot = onehot.cuda()

        face_out = face_net.get_face_logit(face)
        ocular_out = ocular_net.get_ocular_logit(ocular)

        face_out = torch.softmax(face_out, dim=1)
        ocular_out = torch.softmax(ocular_out, dim=1)

        face_mat[i * batch_size:i * batch_size + nof_img, :] = face_out.detach().clone()
        ocular_mat[i * batch_size:i * batch_size + nof_img, :] = ocular_out.detach().clone()


    print('forwarding done')

    p = torch.zeros((1054,1054)).type(torch.float32).cuda()
    for i in range(nof_valset):
        p += face_mat[i].view(1054, 1) * ocular_mat[i].view(1,1054)
    p /= nof_valset
    p = ((p + p.t()) / 2)
    h_x = (-p.sum(dim=1) * torch.log(p.sum(dim=1)+eps)).sum()
    h_y = (-p.sum(dim=0) * torch.log(p.sum(dim=0)+eps)).sum()
    pi = p.sum(dim=1).view(1054, 1).expand(1054, 1054)
    pj = p.sum(dim=0).view(1, 1054).expand(1054, 1054)

    mutual_info = (p * (torch.log(p + eps) - torch.log(pi + eps) - torch.log(pj + eps) ) ).sum()
    normalized_mutual_info = 2 * mutual_info / (h_x + h_y)

    print(mutual_info)
    print(normalized_mutual_info)

