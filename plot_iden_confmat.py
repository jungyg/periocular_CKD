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

def confusion_matrix(feature1, feature2, label1, label2, num_classes):

    dist = torch.matmul(feature1, feature2.t())
    dist = dist.cpu().numpy()

    confusion_mat = np.zeros((num_classes, num_classes))
    confusion_bin = np.zeros((num_classes, num_classes))


    for i in range(len(label1)):
        for j in range(len(label2)):
            class_i, class_j = label1[i], label2[j]
            confusion_mat[class_i, class_j] += dist[i,j]
            confusion_bin[class_i, class_j] += 1.
    confusion_mat /= confusion_bin

    return confusion_mat



torch.backends.cudnn.enabled = True
torch.cuda.set_device(0)

exp_list = ['01', '02', '04', '05', '06', '07']
epoch = '70'

for exp in exp_list:
    print(exp)
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
    elif exp == '07':
        model = networks.F_net(num_classes=1054)
    ## load model
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model_'+epoch+'.pth.tar'), map_location='cpu')['state_dict'])
    model = model.cuda()
    model.eval()

    cmc_dict = {}
    rank1_dict = {}

    dset_list = ['ethnic', 'pubfig', 'ar']

    for dset in dset_list:
        print(dset + ' dataset identification start')
        dset_type = 'probe' if dset == 'ethnic' else 'gallery'

        bench_dataset = dload.benchmark_dataset(dset_name=dset, dset_type=dset_type)
        bench_loader = torch.utils.data.DataLoader(bench_dataset, batch_size=batch_size, num_workers=4)
        nof_dataset = len(bench_dataset)
        num_classes = bench_dataset.nof_identity
        embedding_mat = torch.zeros((nof_dataset, embedding_size)).cuda()
        label_list = []
        with torch.no_grad():
            for i, (face, ocular, label, onehot) in enumerate(bench_loader):
                nof_img = ocular.shape[0]
                ocular = ocular.cuda()
                face = face.cuda()
                onehot = onehot.cuda()
                if exp == '07':
                    feature = model.get_face_feature(face)
                else:
                    feature = model.get_ocular_feature(ocular)

                feature /= torch.norm(feature, dim=1, keepdim=True)

                embedding_mat[i * batch_size:i * batch_size + nof_img, :] = feature.detach().clone()
                label_list.extend(label.numpy())

        label_list = np.array(label_list)
        confusion_mat = confusion_matrix(embedding_mat, embedding_mat, label_list, label_list, num_classes)

        if not os.path.exists('./iden_all_confusion_mats/exp'+exp):
            os.makedirs('./iden_all_confusion_mats/exp'+exp)
        np.save('./iden_all_confusion_mats/exp'+exp+'/'+dset+'_mat.npy', confusion_mat)

