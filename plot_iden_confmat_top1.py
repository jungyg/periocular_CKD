import os
import time
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import networks as networks
import shared_networks as shared_networks
from collections import OrderedDict
import matplotlib.pyplot as plt
import dload
from utils import *
import json
import torchvision.transforms as transforms
import torch.utils.data as data
from sklearn.metrics import roc_auc_score, plot_roc_curve, roc_curve
from scipy import interp
from PIL import Image

torch.backends.cudnn.enabled = True
torch.cuda.set_device(0)

exp = '01'
epoch = '70'

root_dir = os.path.join('results', 'exp_'+exp)
model_dir = os.path.join(root_dir, 'models')
figure_dir = os.path.join(root_dir, 'figures', epoch, 'ocular')

makedir(root_dir)
makedir(model_dir)
makedir(figure_dir)

batch_size = 500
embedding_size = 512
nof_train_iden = 1054

model = shared_networks.shared_network(num_classes=1054)
if exp == '05' or exp == '06' or exp=='15':
    model = networks.O_net(num_classes=1054)
## load model
model.load_state_dict(torch.load(os.path.join(model_dir, 'model_'+epoch+'.pth.tar'), map_location='cpu')['state_dict'])
model = model.cuda()
model.eval()

cmc_dict = {}
rank1_dict = {}


dset_list = ['ethnic', 'pubfig', 'facescrub', 'imdb_wiki', 'ar', 'ytf']


for dset in dset_list:
    print(dset + ' dataset identification start')
    if dset == 'ethnic':
        dset_list = ['probe']
    elif dset == 'pubfig':
        dset_list = ['probe1', 'probe2', 'probe3']
    elif dset == 'facescrub':
        dset_list = ['probe1', 'probe2']
    elif dset == 'imdb_wiki':
        dset_list = ['probe1', 'probe2', 'probe3']
    elif dset == 'ar':
        dset_list = ['blur', 'exp_illum', 'occlude', 'scarf']
    elif dset == 'ytf':
        dset_list = ['probe']

    galleryset = dload.benchmark_dataset(dset_name=dset, dset_type='gallery')
    gallery_loader = torch.utils.data.DataLoader(galleryset, batch_size=batch_size, num_workers=4)
    nof_galleryset = len(galleryset)
    nof_benchmark_iden = galleryset.nof_identity
    confusion_mat = torch.zeros((nof_benchmark_iden, nof_benchmark_iden)).cuda()
    confusion_bin = torch.zeros((nof_benchmark_iden, nof_benchmark_iden)).cuda()
    gallery_embedding_mat = torch.zeros((nof_galleryset, embedding_size)).cuda()
    gallery_label_list = []
    with torch.no_grad():
        for i, (face, ocular, label, onehot) in enumerate(gallery_loader):
            nof_img = ocular.shape[0]
            ocular = ocular.cuda()
            onehot = onehot.cuda()
            feature = model.get_ocular_feature(ocular)

            gallery_embedding_mat[i * batch_size:i * batch_size + nof_img, :] = feature.detach().clone()
            gallery_label_list.extend(label)

    gallery_embedding_mat /= torch.norm(gallery_embedding_mat, dim=1, keepdim=True)

    for dset_type in dset_list:
        probeset = dload.benchmark_dataset(dset_name=dset, dset_type=dset_type)
        probe_loader = torch.utils.data.DataLoader(probeset, batch_size=batch_size, num_workers=4)
        nof_probeset = len(probeset)
        with torch.no_grad():
            for i, (face, ocular, label, onehot) in enumerate(probe_loader):
                nof_img = ocular.shape[0]
                ocular = ocular.cuda()
                onehot = onehot.cuda()
                # probe_embedding_mat = torch.zeros((nof_img, embedding_size)).cuda()
                # probe_onehot_label = torch.zeros((nof_img, nof_benchmark_iden)).cuda()

                feature = model.get_ocular_feature(ocular).detach().clone()

                feature /= torch.norm(feature, dim=1, keepdim=True)

                cos_dist = torch.matmul(feature, gallery_embedding_mat.t())

                max_val, max_idx = torch.max(cos_dist, dim=1)

                for i in range(nof_img):
                    idx = gallery_label_list[max_idx[i].item()]
                    confusion_mat[label[i], idx] += max_val[i]
                    confusion_bin[label[i], idx] += 1
    
    confusion_mat = confusion_mat / confusion_bin
    for i in range(len(confusion_bin)):
        for j in range(len(confusion_bin)):
            if confusion_bin[i,j] == 0:
                confusion_mat[i,j] = -1
    confusion_mat = confusion_mat.cpu().numpy()

    if not os.path.exists('./iden_top1_confusion_mats/exp'+exp):
        os.makedirs('./iden_top1_confusion_mats/exp'+exp)
    np.save('./iden_top1_confusion_mats/exp'+exp+'/'+dset+'_mat.npy', confusion_mat)

