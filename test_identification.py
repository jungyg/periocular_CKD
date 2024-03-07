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
from models import * 
from collections import OrderedDict
import matplotlib.pyplot as plt
import dataset.face_ocular as face_ocular
from helper.util import *
import json
import torchvision.transforms as transforms
import torch.utils.data as data
from sklearn.metrics import roc_auc_score, plot_roc_curve, roc_curve
from scipy import interp
from PIL import Image
from config import config as cfg
from importlib import import_module



def identification():
            
    cfg.update_with_yaml("ckd.yaml")
    cfg.freeze()

    torch.backends.cudnn.enabled = True
    # torch.cuda.set_device(cfg.device)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device)

    batch_size = 4
    embedding_size = 512
    num_classes = 1054

    root_dir = os.path.join(cfg.result_path, f'DISTILL-{cfg.distill}-NET-{cfg.network}-BN-{cfg.batchnorm}-DSET-{cfg.dataset}')
    model_dir = os.path.join(root_dir, 'models')
    log_dir = os.path.join(root_dir, 'log')

    makedir(root_dir)
    makedir(model_dir)
    makedir(log_dir)

    print('>>>> loading module')
    module = import_module('models.'+cfg.network)
    model = getattr(module, cfg.network)().cuda()


    model.load_state_dict(torch.load(os.path.join(model_dir, 'best_checkpoint.pth'), map_location='cpu')['state_dict'])
    model = model.cuda()
    model.eval()

    cmc_dict = {}
    rank1_dict = {}
    dset_list = ['ethnic', 'pubfig', 'facescrub', 'imdb_wiki', 'ar', 'ytf']

    for dset in dset_list:
        print(dset + ' dataset identification start')
        if dset == 'ethnic':
            dset_list = ['gallery', 'probe']
        elif dset == 'pubfig':
            dset_list = ['gallery', 'probe1', 'probe2', 'probe3']
        elif dset == 'facescrub':
            dset_list = ['gallery', 'probe1', 'probe2']
        elif dset == 'imdb_wiki':
            dset_list = ['gallery', 'probe1', 'probe2', 'probe3']
        elif dset == 'ar':
            dset_list = ['gallery', 'blur', 'exp_illum', 'occlude', 'scarf']
        elif dset == 'ytf':
            dset_list = ['gallery', 'probe']

        num_set = len(dset_list)
        gallery_idx = list(range(num_set)) * num_set
        probe_idx = []
        for i in range(num_set):
            probe_idx += [i]*num_set

        cmc_sum = np.zeros(20)
        cnt = 0
        for i, j in zip(gallery_idx, probe_idx):
            if i == j:
                continue
            cnt += 1
            galleryset = face_ocular.benchmark_dataset(dset_name=dset, dset_type=dset_list[i])
            gallery_loader = torch.utils.data.DataLoader(galleryset, batch_size=batch_size, num_workers=4)

            num_gallery = len(galleryset)
            num_classes = galleryset.num_classes
            gallery_embedding_mat = torch.zeros((num_gallery, embedding_size)).cuda()
            gallery_onehot_label = torch.zeros((num_gallery, num_classes)).cuda()
            with torch.no_grad():
                for i, (face, ocular, label, onehot) in enumerate(gallery_loader):
                    if cfg.network == 'face_network':
                        face = face.cuda()
                        feature = model.get_face_feature(face)
                    else:
                        ocular = ocular.cuda()
                        feature = model.get_ocular_feature(ocular)

                    num_img = ocular.shape[0]
                    onehot = onehot.cuda()

                    gallery_embedding_mat[i * batch_size:i * batch_size + num_img, :] = feature.detach().clone()
                    gallery_onehot_label[i * batch_size:i * batch_size + num_img, :] = onehot

            probeset = face_ocular.benchmark_dataset(dset_name=dset, dset_type=dset_list[j])
            probe_loader = torch.utils.data.DataLoader(probeset, batch_size=batch_size, num_workers=4)
            num_probe = len(probeset)
            with torch.no_grad():
                cmc = [0] * 20
                for i, (face, ocular, label, onehot) in enumerate(probe_loader):
                    if cfg.network == 'face_network':
                        face = face.cuda()
                        feature = model.get_face_feature(face)
                    else:
                        ocular = ocular.cuda()
                        feature = model.get_ocular_feature(ocular)

                    num_img = ocular.shape[0]
                    onehot = onehot.cuda()

                    probe_embedding_mat = torch.zeros((num_img, embedding_size)).cuda()
                    probe_onehot_label = torch.zeros((num_img, num_classes)).cuda()

                    feature = model.get_ocular_feature(ocular)

                    probe_embedding_mat[:num_img, :] = feature.detach().clone()
                    probe_onehot_label[:num_img, :] = onehot
                    ### cmc
                    x_range, tmp_cmc = calculate_cmc(gallery_embedding_mat, probe_embedding_mat,
                                                    gallery_onehot_label, probe_onehot_label)
                    cmc += tmp_cmc * num_img

                cmc /= num_probe
                cmc_sum += cmc

        cmc = cmc_sum / cnt

        plt.plot(x_range, cmc, label='cmc')
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(log_dir, dset + '_cmc.jpg'), bbox_inches='tight')
        plt.close()

        cmc_dict[dset] = list(cmc)
        rank1_dict[dset] = cmc[0]

    print(cfg)

    for k,v in rank1_dict.items():
        print('%.3f' % (v * 100.0))

    with open(os.path.join(log_dir, 'cmc_dict.pkl'), 'wb') as f:
        pickle.dump(cmc_dict, f)

    for k,v in cmc_dict.items():
        cmc_dict[k] = str(v)

    with open(os.path.join(log_dir, 'cmc_json.json'), 'w') as f:
        json.dump(cmc_dict, f, indent=2)

    with open(os.path.join(log_dir, 'rank1_dict.pkl'), 'wb') as f:
        pickle.dump(rank1_dict, f)

    for k,v in rank1_dict.items():
        rank1_dict[k] = str(v)

    with open(os.path.join(log_dir, 'rank1_json.json'), 'w') as f:
        json.dump(rank1_dict, f, indent=2)



if __name__ == '__main__':
    identification()

