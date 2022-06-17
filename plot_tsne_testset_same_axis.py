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
from sklearn.manifold import TSNE
import seaborn as sns


class tsne_dataset(data.Dataset):
    def __init__(self, dset_type, num_classes):
        dset = 'trainingdb'
        assert dset_type in ['val', 'test']
        self.type = dset_type
        self.face_root_dir = os.path.join('/home/yoon/data/face_ocular', dset, dset_type)
        self.ocular_root_dir = os.path.join('/home/yoon/data/face_ocular', dset, dset_type)
        self.total_classes = len(os.listdir(self.face_root_dir))
        self.face_img_dir_list = []
        self.ocular_img_dir_list = []
        self.label_list = []
        self.label_dict = {}

        class_idx_list = np.random.randint(low=0, high=self.total_classes, size=num_classes)
        print(class_idx_list)
        
        cnt = 0
        label = 0
        for iden in sorted(os.listdir(self.face_root_dir)):
            if cnt in class_idx_list:
                self.label_dict[label] = iden
                for img in sorted(os.listdir(os.path.join(self.face_root_dir, iden, 'face'))):
                    self.face_img_dir_list.append(os.path.join(self.face_root_dir, iden, 'face', img))
                    self.label_list.append(label)
                for img in sorted(os.listdir(os.path.join(self.ocular_root_dir, iden, 'periocular'))):
                    self.ocular_img_dir_list.append(os.path.join(self.ocular_root_dir, iden, 'periocular', img))
                label += 1
            cnt += 1

        print(label)

        self.onehot_label = np.zeros((len(self.face_img_dir_list), num_classes))
        for i in range(len(self.face_img_dir_list)):
            self.onehot_label[i, self.label_list[i]] = 1

        self.face_transform = transforms.Compose([transforms.Resize(128),
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
        face = Image.open(self.face_img_dir_list[idx])
        face = self.face_transform(face)
        ocular = Image.open(self.ocular_img_dir_list[idx])
        ocular = self.ocular_transform(ocular)
        label = self.label_list[idx]
        onehot = self.onehot_label[idx]
        return face, ocular, label, onehot


torch.backends.cudnn.enabled = True
torch.cuda.set_device(0)

eps = 1e-5
batch_size = 500

num_classes = 6
palette = np.array(sns.color_palette("hls", num_classes))

print('>>>> loading training dataset')
dataset = tsne_dataset(dset_type='test', num_classes=num_classes)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4)
nof_dataset = len(dataset)
print('{} validating images loaded'.format(nof_dataset))

total_feature_mat = np.zeros((nof_dataset*5, 512)).astype(np.float32)
total_logit_mat = np.zeros((nof_dataset*5, 1054)).astype(np.float32)

## vanilla:
print('CE')
face_net = networks.F_net(num_classes=1054).cuda()
ocular_net = networks.O_net(num_classes=1054).cuda()
face_state_dict = torch.load('./results/exp_07/models/model_70.pth.tar', map_location='cpu')
ocular_state_dict = torch.load('./results/exp_06/models/model_70.pth.tar', map_location='cpu')
face_net.load_state_dict(face_state_dict['state_dict'])
ocular_net.load_state_dict(ocular_state_dict['state_dict'])
face_net.eval()
ocular_net.eval()
face_feature_mat = torch.zeros((nof_dataset, 512), dtype=torch.float32).cuda()
ocular_feature_mat = torch.zeros((nof_dataset, 512), dtype=torch.float32).cuda()
face_logit_mat = torch.zeros((nof_dataset, 1054), dtype=torch.float32).cuda()
ocular_logit_mat = torch.zeros((nof_dataset, 1054), dtype=torch.float32).cuda()
with torch.no_grad():
    for i, (face, ocular, label, onehot) in enumerate(data_loader):
        nof_img, fc, fh, fw = face.size()
        face = face.cuda()
        ocular = ocular.cuda()

        face_feature, face_logit = face_net(face)
        ocular_feature, ocular_logit = ocular_net(ocular)

        face_feature_mat[i*nof_img : (i+1)*nof_img, :] = face_feature.detach()
        ocular_feature_mat[i*nof_img : (i+1)*nof_img, :] = ocular_feature.detach()
        face_logit_mat[i*nof_img : (i+1)*nof_img, :] = face_logit.detach()
        ocular_logit_mat[i*nof_img : (i+1)*nof_img, :] = ocular_logit.detach()




face_feature_mat = face_feature_mat.cpu().numpy()
ocular_feature_mat = ocular_feature_mat.cpu().numpy()
face_logit_mat = face_logit_mat.cpu().numpy()
ocular_logit_mat = ocular_logit_mat.cpu().numpy()

total_feature_mat[:nof_dataset,:] = face_feature_mat
total_feature_mat[nof_dataset:2*nof_dataset,:] = ocular_feature_mat
total_logit_mat[:nof_dataset,:] = face_logit_mat
total_logit_mat[nof_dataset:2*nof_dataset,:] = ocular_logit_mat



## KD:
print('KD')
ocular_net = networks.O_net(num_classes=1054).cuda()
ocular_state_dict = torch.load('./results/exp_99/models/model_70.pth.tar', map_location='cpu')
ocular_net.load_state_dict(ocular_state_dict['state_dict'])
ocular_net.eval()
ocular_feature_mat = torch.zeros((nof_dataset, 512), dtype=torch.float32).cuda()
ocular_logit_mat = torch.zeros((nof_dataset, 1054), dtype=torch.float32).cuda()

with torch.no_grad():
    for i, (face, ocular, label, onehot) in enumerate(data_loader):
        nof_img, fc, fh, fw = face.size()
        ocular = ocular.cuda()

        ocular_feature, ocular_logit = ocular_net(ocular)

        ocular_feature_mat[i*nof_img : (i+1)*nof_img, :] = ocular_feature.detach()
        ocular_logit_mat[i*nof_img : (i+1)*nof_img, :] = ocular_logit.detach()



ocular_feature_mat = ocular_feature_mat.cpu().numpy()
ocular_logit_mat = ocular_logit_mat.cpu().numpy()

total_feature_mat[2*nof_dataset:3*nof_dataset,:] = ocular_feature_mat
total_logit_mat[2*nof_dataset:3*nof_dataset,:] = ocular_logit_mat


print('CKD')
model = shared_networks.shared_network(num_classes=1054).cuda()
model.load_state_dict(torch.load('./results/exp_01/models/model_70.pth.tar', map_location='cpu')['state_dict'])
model.eval()
face_feature_mat = torch.zeros((nof_dataset, 512), dtype=torch.float32).cuda()
ocular_feature_mat = torch.zeros((nof_dataset, 512), dtype=torch.float32).cuda()
face_logit_mat = torch.zeros((nof_dataset, 1054), dtype=torch.float32).cuda()
ocular_logit_mat = torch.zeros((nof_dataset, 1054), dtype=torch.float32).cuda()
with torch.no_grad():
    for i, (face, ocular, label, onehot) in enumerate(data_loader):
        nof_img, fc, fh, fw = face.size()
        face = face.cuda()
        ocular = ocular.cuda()

        # face_feature, ocular_feature, face_out, ocular_out = model(face, ocular)
        face_feature, face_logit = model.face_forward(face)
        ocular_feature, ocular_logit = model.ocular_forward(ocular)

        face_feature_mat[i*nof_img : (i+1)*nof_img, :] = face_feature.detach()
        face_logit_mat[i*nof_img : (i+1)*nof_img, :] = face_logit.detach()
        ocular_feature_mat[i*nof_img : (i+1)*nof_img, :] = ocular_feature.detach()
        ocular_logit_mat[i*nof_img : (i+1)*nof_img, :] = ocular_logit.detach()

ocular_feature_mat = ocular_feature_mat.cpu().numpy()
ocular_logit_mat = ocular_logit_mat.cpu().numpy()
face_feature_mat = face_feature_mat.cpu().numpy()
face_logit_mat = face_logit_mat.cpu().numpy()

print(np.sum(ocular_logit_mat-face_logit_mat))

total_feature_mat[3*nof_dataset:4*nof_dataset,:] = face_feature_mat
total_logit_mat[3*nof_dataset:4*nof_dataset,:] = face_logit_mat
total_feature_mat[4*nof_dataset:5*nof_dataset,:] = ocular_feature_mat
total_logit_mat[4*nof_dataset:5*nof_dataset,:] = ocular_logit_mat

sns.set_theme()
total_tsne = TSNE(n_components=2, learning_rate='auto', random_state=0).fit_transform(total_feature_mat)
label = label.cpu().numpy()

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 4))

ax1.scatter(total_tsne[:nof_dataset, 0], total_tsne[:nof_dataset, 1], c=palette[label.astype(np.int)], s=3)
ax1.set_title('face')
ax2.scatter(total_tsne[nof_dataset:nof_dataset*2, 0], total_tsne[nof_dataset:nof_dataset*2, 1], c=palette[label.astype(np.int)], s=3)
ax2.set_title('ocular')
ax3.scatter(total_tsne[nof_dataset*2:nof_dataset*3, 0], total_tsne[nof_dataset*2:nof_dataset*3, 1], c=palette[label.astype(np.int)], s=3)
ax3.set_title('kd')
ax4.scatter(total_tsne[nof_dataset*3:nof_dataset*4, 0], total_tsne[nof_dataset*3:nof_dataset*4, 1], c=palette[label.astype(np.int)], s=3)
ax4.set_title('ckd face')
ax5.scatter(total_tsne[nof_dataset*4:, 0], total_tsne[nof_dataset*4:, 1], c=palette[label.astype(np.int)], s=3)
ax5.set_title('ckd')
plt.savefig('./tsne_agg_feature.jpg')
plt.close()

total_tsne2 = TSNE(n_components=2, learning_rate='auto', random_state=0).fit_transform(total_logit_mat)
sns.set_theme()

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 4))

ax1.scatter(total_tsne2[:nof_dataset, 0], total_tsne2[:nof_dataset, 1], c=palette[label.astype(np.int)], s=3)
ax1.set_title('face')
ax2.scatter(total_tsne2[nof_dataset:nof_dataset*2, 0], total_tsne2[nof_dataset:nof_dataset*2, 1], c=palette[label.astype(np.int)], s=3)
ax2.set_title('ocular')
ax3.scatter(total_tsne2[nof_dataset*2:nof_dataset*3, 0], total_tsne2[nof_dataset*2:nof_dataset*3, 1], c=palette[label.astype(np.int)], s=3)
ax3.set_title('kd')
ax4.scatter(total_tsne2[nof_dataset*3:nof_dataset*4, 0], total_tsne2[nof_dataset*3:nof_dataset*4, 1], c=palette[label.astype(np.int)], s=3)
ax4.set_title('ckd face')
ax5.scatter(total_tsne2[nof_dataset*4:, 0], total_tsne2[nof_dataset*4:, 1], c=palette[label.astype(np.int)], s=3)
ax5.set_title('ckd')
plt.savefig('./tsne_agg_logit.jpg')
plt.close()



