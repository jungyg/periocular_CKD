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





def compute_eer(fpr,tpr):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    fnr = 1-tpr
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))
    return eer



def verification():

    cfg.update_with_yaml("ckd.yaml")
    cfg.freeze()

    torch.backends.cudnn.enabled = True
    # torch.cuda.set_device(cfg.device)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device)


    batch_size = 500
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

    eer_dict = {'val':0, 'ethnic':0, 'pubfig':0, 'facescrub':0,'wiki':0, 'ar': 0, 'ytf':0}

    dset_list = ['ethnic', 'pubfig', 'facescrub', 'imdb_wiki', 'ar', 'ytf']

    ver_img_per_class = 4

    for dset_name in dset_list:
        if dset_name == 'ethnic':
            dset = face_ocular.verification_dataset(dset=dset_name, dset_type='probe', ver_img_per_class=ver_img_per_class)
        else:
            dset = face_ocular.verification_dataset(dset=dset_name, dset_type='gallery', ver_img_per_class=ver_img_per_class)
        dloader = torch.utils.data.DataLoader(dset, batch_size=batch_size, num_workers=4)
        num_dset = len(dset)
        num_classes = dset.num_classes
        embedding_mat = torch.zeros((num_dset, embedding_size)).cuda()
        label_mat = torch.zeros((num_dset, num_classes)).cuda()

        with torch.no_grad():
            for i, (face, ocular, label, onehot) in enumerate(dloader):
                num_img = ocular.shape[0]
                ocular = ocular.cuda()
                onehot = onehot.cuda()

                feature = model.get_ocular_feature(ocular)

                embedding_mat[i*batch_size:i*batch_size+num_img, :] = feature.detach().clone()
                label_mat[i*batch_size:i*batch_size+num_img, :] = onehot

            ### roc
            embedding_mat /= torch.norm(embedding_mat, p=2, dim=1, keepdim=True)

            score_mat = torch.matmul(embedding_mat, embedding_mat.t()).cpu()
            gen_mat = torch.matmul(label_mat, label_mat.t()).cpu()
            gen_r, gen_c = torch.where(gen_mat == 1)
            imp_r, imp_c = torch.where(gen_mat == 0)

            gen_score = score_mat[gen_r, gen_c].cpu().numpy()
            imp_score = score_mat[imp_r, imp_c].cpu().numpy()

            y_gen = np.ones(gen_score.shape[0])
            y_imp = np.zeros(imp_score.shape[0])

            score = np.concatenate((gen_score, imp_score))
            y = np.concatenate((y_gen, y_imp))


            fpr_tmp, tpr_tmp, _ = roc_curve(y, score)
            eer_dict[dset_name] = compute_eer(fpr_tmp, tpr_tmp)


    print(cfg)

    for k,v in eer_dict.items():
        print('%.3f' % (v * 100.0))

    with open(os.path.join(log_dir, 'eer_dict.pkl'), 'wb') as f:
        pickle.dump(eer_dict, f)

    for k,v in eer_dict.items():
        eer_dict[k] = str(v)

    with open(os.path.join(log_dir, 'eer_dict.json'), 'w') as f:
        json.dump(eer_dict, f, indent=2)












if __name__ == '__main__':
    verification()

