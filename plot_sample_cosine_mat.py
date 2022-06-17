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
import seaborn as sns

torch.backends.cudnn.enabled = True
torch.cuda.set_device(0)

# plt.style.use('classic')

cm = 'RdBu_r'


batch_size = 500
embedding_size = 512
nof_train_iden = 1054

# sns.set_palette("colorblind")
# sns.set_theme()

width = 0.2
margin = 0.015
font_size = 16
color_palette = [(94, 190, 171), (230, 161, 118), (0, 103, 138), (205, 205, 205)]
color_palette = np.array(color_palette) / 255

bins = 10
bins = np.linspace(-0.3, 0.3, 300)

print('CE')
dset_list = ['ar']
for dset in dset_list:
    fig, axs = plt.subplots(3,3, figsize=(20,20))

    print(dset + ' dataset identification start')
    face_net = networks.F_net(num_classes=1054).cuda()
    ocular_net = networks.O_net(num_classes=1054).cuda()
    face_state_dict = torch.load('./results/exp_07/models/model_70.pth.tar', map_location='cpu')
    ocular_state_dict = torch.load('./results/exp_06/models/model_70.pth.tar', map_location='cpu')
    face_net.load_state_dict(face_state_dict['state_dict'])
    ocular_net.load_state_dict(ocular_state_dict['state_dict'])
    face_net.eval()
    ocular_net.eval()
    dataset = dload.benchmark_dataset(dset_name=dset, dset_type='gallery')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4)
    nof_dataset = len(dataset)
    nof_iden = dataset.nof_identity
    face_mat = torch.zeros((nof_dataset, embedding_size)).cuda()
    ocular_mat = torch.zeros((nof_dataset, embedding_size)).cuda()
    with torch.no_grad():
        for i, (face, ocular, label, onehot) in enumerate(dataloader):
            nof_img = ocular.shape[0]
            ocular = ocular.cuda()
            face = face.cuda()
            
            face_feature = face_net.get_face_feature(face)
            ocular_feature = ocular_net.get_ocular_feature(ocular)

            face_mat[i * batch_size:i * batch_size + nof_img, :] = face_feature.detach().clone()
            ocular_mat[i * batch_size:i * batch_size + nof_img, :] = ocular_feature.detach().clone()

        face_mat /= torch.norm(face_mat, dim=1, keepdim=True)
        ocular_mat /= torch.norm(ocular_mat, dim=1, keepdim=True)

        face_inner = torch.matmul(face_mat, face_mat.t())
        ocular_inner = torch.matmul(ocular_mat, ocular_mat.t())

        out = (face_inner - ocular_inner) / 2.0

        face_inner = face_inner.cpu().numpy()
        ocular_inner = ocular_inner.cpu().numpy()
        out = out.cpu().numpy()
        print(np.mean(out))

        np.save('./CE_face_inner.npy', face_inner)
        np.save('./CE_ocular_inner.npy', ocular_inner)
        np.save('./CE_diff.npy', out)

        axs[0,0].imshow(face_inner, cmap=cm)
        axs[1,0].imshow(ocular_inner, cmap=cm)
        axs[2,0].imshow(out, cmap=cm)

        # plt.colorbar()
        # plt.savefig(f'sample_cosine_CE_{dset}.jpg')
        # plt.close()

    print('KD')
    face_net = networks.F_net(num_classes=1054).cuda()
    ocular_net = networks.O_net(num_classes=1054).cuda()
    face_state_dict = torch.load('./results/exp_07/models/model_70.pth.tar', map_location='cpu')
    ocular_state_dict = torch.load('./results/exp_99/models/model_70.pth.tar', map_location='cpu')
    face_net.load_state_dict(face_state_dict['state_dict'])
    ocular_net.load_state_dict(ocular_state_dict['state_dict'])
    face_net.eval()
    ocular_net.eval()
    face_mat = torch.zeros((nof_dataset, embedding_size)).cuda()
    ocular_mat = torch.zeros((nof_dataset, embedding_size)).cuda()
    with torch.no_grad():
        for i, (face, ocular, label, onehot) in enumerate(dataloader):
            nof_img = ocular.shape[0]
            ocular = ocular.cuda()
            face = face.cuda()
            
            face_feature = face_net.get_face_feature(face)
            ocular_feature = ocular_net.get_ocular_feature(ocular)

            face_mat[i * batch_size:i * batch_size + nof_img, :] = face_feature.detach().clone()
            ocular_mat[i * batch_size:i * batch_size + nof_img, :] = ocular_feature.detach().clone()

        face_mat /= torch.norm(face_mat, dim=1, keepdim=True)
        ocular_mat /= torch.norm(ocular_mat, dim=1, keepdim=True)

        face_inner = torch.matmul(face_mat, face_mat.t())
        ocular_inner = torch.matmul(ocular_mat, ocular_mat.t())

        out = (face_inner - ocular_inner) / 2.0


        face_inner = face_inner.cpu().numpy()
        ocular_inner = ocular_inner.cpu().numpy()
        out = out.cpu().numpy()
        print(np.mean(out))


        np.save('./KD_face_inner.npy', face_inner)
        np.save('./KD_ocular_inner.npy', ocular_inner)
        np.save('./KD_diff.npy', out)


        axs[0,1].imshow(face_inner, cmap=cm)
        axs[1,1].imshow(ocular_inner, cmap=cm)
        axs[2,1].imshow(out, cmap=cm)
        # plt.colorbar()
        # plt.savefig(f'sample_cosine_KD_{dset}.jpg')
        # plt.close()

    print('CKD')
    model = shared_networks.shared_network(num_classes=1054).cuda()
    model.load_state_dict(torch.load('./results/exp_01/models/model_70.pth.tar', map_location='cpu')['state_dict'])
    model.eval()
    face_mat = torch.zeros((nof_dataset, embedding_size)).cuda()
    ocular_mat = torch.zeros((nof_dataset, embedding_size)).cuda()
    with torch.no_grad():
        for i, (face, ocular, label, onehot) in enumerate(dataloader):
            nof_img = ocular.shape[0]
            ocular = ocular.cuda()
            face = face.cuda()
            
            face_feature = model.get_face_feature(face)
            ocular_feature = model.get_ocular_feature(ocular)

            face_mat[i * batch_size:i * batch_size + nof_img, :] = face_feature.detach().clone()
            ocular_mat[i * batch_size:i * batch_size + nof_img, :] = ocular_feature.detach().clone()

        face_mat /= torch.norm(face_mat, dim=1, keepdim=True)
        ocular_mat /= torch.norm(ocular_mat, dim=1, keepdim=True)

        face_inner = torch.matmul(face_mat, face_mat.t())
        ocular_inner = torch.matmul(ocular_mat, ocular_mat.t())

        out = (face_inner - ocular_inner) / 2.0


        face_inner = face_inner.cpu().numpy()
        ocular_inner = ocular_inner.cpu().numpy()
        out = out.cpu().numpy()
        print(np.mean(out))


        np.save('./CKD_face_inner.npy', face_inner)
        np.save('./CKD_ocular_inner.npy', ocular_inner)
        np.save('./CKD_diff.npy', out)


        axs[0,2].imshow(face_inner, cmap=cm)
        im = axs[1,2].imshow(ocular_inner, cmap=cm)
        axs[2,2].imshow(out, cmap=cm)


        ## Face Entropy
        # ax8.bar_label(rects1, padding=3, fontsize=9)
        # ax8.bar_label(rects2, padding=3, fontsize=9)
        # ax8.bar_label(rects3, padding=3, fontsize=9)
        # ax8.bar_label(rects4, padding=3, fontsize=9)


        # Add some text for labels, title and custom x-axis tick labels, etc.
        # ax2.set_ylabel('Scores')
        # axs.tick_params(axis='both', which='major', labelsize=font_size)


        # ax2.set_title('(a)', fontsize=20)
        # ax4.set_title('(b)', fontsize=20)
        # ax5.set_title('(c)', fontsize=20)

        font_size = 16

        axs[0,0].tick_params(axis='both', which='major', labelsize=font_size)
        axs[0,1].tick_params(axis='both', which='major', labelsize=font_size)
        axs[0,2].tick_params(axis='both', which='major', labelsize=font_size)
        axs[1,0].tick_params(axis='both', which='major', labelsize=font_size)
        axs[1,1].tick_params(axis='both', which='major', labelsize=font_size)
        axs[1,2].tick_params(axis='both', which='major', labelsize=font_size)
        axs[2,0].tick_params(axis='both', which='major', labelsize=font_size)
        axs[2,1].tick_params(axis='both', which='major', labelsize=font_size)
        axs[2,2].tick_params(axis='both', which='major', labelsize=font_size)


        fig.tight_layout()
        cbar = plt.colorbar(im, ax=axs,shrink=0.5)
        cbar.ax.tick_params(labelsize=16)
        plt.savefig(f'plot_cosine_final_{dset}.pdf')
        plt.close()

