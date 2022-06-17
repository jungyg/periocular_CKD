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




def kl_div_half(logit1, logit2):
    ## distances are sum of batches
    loss = 0.0
    loss += F.kl_div(F.log_softmax(logit1, dim=1), F.softmax(logit2, dim=1), reduction="batchmean") / 2.0
    loss += F.kl_div(F.log_softmax(logit2, dim=1), F.softmax(logit1, dim=1), reduction="batchmean") / 2.0

    if len(logit1.shape) == 3:
        loss *= logit1.shape[0]

    return loss


def js_div(logit1, logit2):
    ## distances are sum of batches
    prob1 = F.softmax(logit1, dim=1)
    prob2 = F.softmax(logit2, dim=1)
    
    total_m = 0.5 * (prob1 + prob2)
    
    loss = 0.0
    loss += F.kl_div(F.log_softmax(logit1, dim=1), total_m, reduction="batchmean") / 2.0
    loss += F.kl_div(F.log_softmax(logit2, dim=1), total_m, reduction="batchmean") / 2.0

    if len(logit1.shape) == 3:
        loss *= logit1.shape[0]

    return loss

def hellinger_dist(logit1, logit2):
    ## distances are sum of batches
    prob1 = F.softmax(logit1, dim=1)
    prob2 = F.softmax(logit2, dim=1)

    hell_dist = torch.sum(torch.norm(torch.sqrt(prob1) - torch.sqrt(prob2), dim=1) / np.sqrt(2))
    return hell_dist

def bhatt_dist(logit1, logit2):
    ## distances are sum of batches

    prob1 = F.softmax(logit1, dim=1)
    prob2 = F.softmax(logit2, dim=1)

    bhatt = torch.sum(torch.sqrt(prob1 * prob2), dim=1)
    bhatt = -torch.log(bhatt)
    bhatt = torch.sum(bhatt)
    return bhatt

def mutual_info(probs1, probs2):
    n, c = probs1.shape
    p_joint = torch.zeros((c,c)).type(torch.float32).cuda()
    for i in range(n):
        p_joint += torch.matmul(probs1[i].view(-1, 1), probs2[i].view(1,-1))
    p_joint /= n
    p_joint_sym = ((p_joint + p_joint.t()) / 2)

    # h_x = (-p_joint_sym.sum(dim=1) * torch.log(p_joint_sym.sum(dim=1)+1e-5)).sum()
    # h_y = (-p_joint_sym.sum(dim=0) * torch.log(p_joint_sym.sum(dim=0)+1e-5)).sum()
    pi = p_joint_sym.sum(dim=0).view(1, -1).expand(c, c)
    pj = p_joint_sym.sum(dim=1).view(-1, 1).expand(c, c)

    mutual_info = torch.sum(p_joint_sym * (torch.log(p_joint_sym/(pi * pj) + 1e-5)))
    # normalized_mutual_info = 2 * mutual_info / (h_x + h_y)
    return mutual_info

    


torch.backends.cudnn.enabled = True
torch.cuda.set_device(1)

eps = 1e-5
batch_size = 128


class train_dataset(data.Dataset):
    def __init__(self, dset_type):
        dset = 'trainingdb'
        self.type = dset_type
        self.face_root_dir = os.path.join('/home/yoon/data/face_ocular', dset, dset_type)
        self.ocular_root_dir = os.path.join('/home/yoon/data/face_ocular', dset, dset_type)
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

        self.ocular_img_dir_list = list(reversed(self.ocular_img_dir_list))

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
valset = train_dataset(dset_type='val')
val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, num_workers=4)
nof_valset = len(valset)
print('{} validating images loaded'.format(nof_valset))

kl_crit = nn.KLDivLoss(reduction='batchmean').cuda()


## vanilla:
face_mat = torch.zeros((nof_valset, 1054)).type(torch.float32).cuda()
ocular_mat = torch.zeros((nof_valset, 1054)).type(torch.float32).cuda()

face_net = networks.F_net(num_classes=1054).cuda()
ocular_net = networks.O_net(num_classes=1054).cuda()
face_state_dict = torch.load('./results/exp_07/models/model_70.pth.tar', map_location='cpu')
ocular_state_dict = torch.load('./results/exp_06/models/model_70.pth.tar', map_location='cpu')
face_net.load_state_dict(face_state_dict['state_dict'])
ocular_net.load_state_dict(ocular_state_dict['state_dict'])
face_net.eval()
ocular_net.eval()
with torch.no_grad():
    half_kl_sum = 0.0
    js_sum = 0.0
    hell_sum = 0.0
    bhatt_sum = 0.0
    for i, (face, ocular, label, onehot) in enumerate(val_loader):
        nof_img, fc, fh, fw = face.size()
        face = face.cuda()
        ocular = ocular.cuda()

        face_out = face_net.get_face_logit(face)
        ocular_out = ocular_net.get_ocular_logit(ocular)

        face_mat[i * batch_size:i * batch_size + nof_img, :] = F.softmax(face_out, dim=1).detach().clone()
        ocular_mat[i * batch_size:i * batch_size + nof_img, :] = F.softmax(ocular_out, dim=1).detach().clone()


        half_kl = kl_div_half(face_out, ocular_out)
        half_kl_sum += half_kl

        js = js_div(face_out, ocular_out)
        js_sum += js

        hell = hellinger_dist(face_out, ocular_out)
        hell_sum += hell

        bhatt = bhatt_dist(face_out, ocular_out)
        bhatt_sum += bhatt

    m_info = mutual_info(face_mat, ocular_mat).cpu().numpy()

    print(f'half KL : {half_kl_sum/nof_valset}, JS : {js_sum/nof_valset}, Hellinger : {hell_sum/nof_valset}, Bhatt : {bhatt_sum/nof_valset}, MI : {m_info}')
    print()


for exp in ['205','01', '02', '04', '05']:
    face_mat = torch.zeros((nof_valset, 1054)).type(torch.float32).cuda()
    ocular_mat = torch.zeros((nof_valset, 1054)).type(torch.float32).cuda()

    half_kl_sum = 0.0
    js_sum = 0.0
    hell_sum = 0.0
    bhatt_sum = 0.0

    print(exp)
    if exp == '05':
        face_net = networks.F_net(num_classes=1054).cuda()
        ocular_net = networks.O_net(num_classes=1054).cuda()
        state_dict = torch.load('./results/exp_'+exp+'/models/model_70.pth.tar', map_location='cpu')
        face_net.load_state_dict(state_dict['face_state_dict'])
        ocular_net.load_state_dict(state_dict['state_dict'])
        face_net.eval()
        ocular_net.eval()
        with torch.no_grad():

            for i, (face, ocular, label, onehot) in enumerate(val_loader):
                nof_img, fc, fh, fw = face.size()
                face = face.cuda()
                ocular = ocular.cuda()

                face_out = face_net.get_face_logit(face)
                ocular_out = ocular_net.get_ocular_logit(ocular)

                face_mat[i * batch_size:i * batch_size + nof_img, :] = F.softmax(face_out, dim=1).detach().clone()
                ocular_mat[i * batch_size:i * batch_size + nof_img, :] = F.softmax(ocular_out, dim=1).detach().clone()

                half_kl = kl_div_half(face_out, ocular_out)
                half_kl_sum += half_kl

                js = js_div(face_out, ocular_out)
                js_sum += js

                hell = hellinger_dist(face_out, ocular_out)
                hell_sum += hell

                bhatt = bhatt_dist(face_out, ocular_out)
                bhatt_sum += bhatt

            m_info = mutual_info(face_mat, ocular_mat).cpu().numpy()

            print(f'half KL : {half_kl_sum/nof_valset}, JS : {js_sum/nof_valset}, Hellinger : {hell_sum/nof_valset}, Bhatt : {bhatt_sum/nof_valset}, MI : {m_info}')
            print()


    else:
        model = shared_networks.shared_network(num_classes=1054).cuda()
        model.load_state_dict(torch.load('./results/exp_'+exp+'/models/model_70.pth.tar', map_location='cpu')['state_dict'])
        model.eval()

        with torch.no_grad():
            for i, (face, ocular, label, onehot) in enumerate(val_loader):
                nof_img, fc, fh, fw = face.size()
                face = face.cuda()
                ocular = ocular.cuda()

                face_out = model.get_face_logit(face)
                ocular_out = model.get_ocular_logit(ocular)

                face_mat[i * batch_size:i * batch_size + nof_img, :] = F.softmax(face_out, dim=1).detach().clone()
                ocular_mat[i * batch_size:i * batch_size + nof_img, :] = F.softmax(ocular_out, dim=1).detach().clone()

                half_kl = kl_div_half(face_out, ocular_out)
                half_kl_sum += half_kl

                js = js_div(face_out, ocular_out)
                js_sum += js

                hell = hellinger_dist(face_out, ocular_out)
                hell_sum += hell

                bhatt = bhatt_dist(face_out, ocular_out)
                bhatt_sum += bhatt

            m_info = mutual_info(face_mat, ocular_mat).cpu().numpy()

            print(f'half KL : {half_kl_sum/nof_valset}, JS : {js_sum/nof_valset}, Hellinger : {hell_sum/nof_valset}, Bhatt : {bhatt_sum/nof_valset}, MI : {m_info}')
            print()
