import torch.utils.data as data
from PIL import Image
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class train_dataset(data.Dataset):
    def __init__(self, dset_type):
        dset = 'trainingdb'
        assert dset_type in ['train', 'val', 'test']
        self.type = dset_type
        self.root_dir = os.path.join('/home/yoon/data/face_ocular', dset, dset_type)
        self.num_classes = len(os.listdir(self.root_dir))
        self.face_img_dir_list = []
        self.ocular_img_dir_list = []
        self.label_list = []
        self.label_dict = {}
        cnt = 0
        for iden in sorted(os.listdir(self.root_dir)):
            self.label_dict[cnt] = iden
            for img in sorted(os.listdir(os.path.join(self.root_dir, iden, 'face'))):
                self.face_img_dir_list.append(os.path.join(self.root_dir, iden, 'face', img))
                self.label_list.append(cnt)
            for img in sorted(os.listdir(os.path.join(self.root_dir, iden, 'periocular'))):
                self.ocular_img_dir_list.append(os.path.join(self.root_dir, iden, 'periocular', img))
            cnt += 1

        self.onehot_label = np.zeros((len(self.face_img_dir_list), self.num_classes))
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



class benchmark_dataset(data.Dataset):
    def __init__(self, dset_name, dset_type):
        dset = dset_name
        assert dset in ['ethnic', 'pubfig', 'facescrub', 'imdb_wiki', 'ar', 'ytf']
        if dset == 'ethnic':
            self.root_dir = os.path.join('/home/yoon/data/face_ocular', dset, 'Recognition', dset_type)
        else:
            self.root_dir = os.path.join('/home/yoon/data/face_ocular', dset, dset_type)
        self.num_classes = len(os.listdir(self.root_dir))
        self.face_img_dir_list = []
        self.ocular_img_dir_list = []
        self.label_list = []
        cnt = 0
        for iden in sorted(os.listdir(self.root_dir)):
            for img in sorted(os.listdir(os.path.join(self.root_dir, iden, 'face'))):
                self.face_img_dir_list.append(os.path.join(self.root_dir, iden, 'face', img))
            for img in sorted(os.listdir(os.path.join(self.root_dir, iden, 'periocular'))):
                self.ocular_img_dir_list.append(os.path.join(self.root_dir, iden, 'periocular', img))
                self.label_list.append(cnt)
            cnt += 1

        self.onehot_label = np.zeros((len(self.ocular_img_dir_list), self.num_classes))
        for i in range(len(self.ocular_img_dir_list)):
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
        return len(self.ocular_img_dir_list)

    def __getitem__(self, idx):
        face = Image.open(self.face_img_dir_list[idx])
        face = self.face_transform(face)
        ocular = Image.open(self.ocular_img_dir_list[idx])
        ocular = self.ocular_transform(ocular)
        label = self.label_list[idx]
        onehot = self.onehot_label[idx]
        return face, ocular, label, onehot



class verification_dataset(data.Dataset):
    def __init__(self, dset, dset_type='gallery', ver_img_per_class=4):
        if dset == 'ethnic':
            self.root_dir = os.path.join('/home/yoon/data/face_ocular', dset, 'Recognition', dset_type)
        else:
            self.root_dir = os.path.join('/home/yoon/data/face_ocular', dset, dset_type)
        self.num_classes = len(os.listdir(self.root_dir))
        self.face_img_dir_list = []
        self.ocular_img_dir_list = []
        self.label_list = []
        self.label_dict = {}
        cnt = 0
        for iden in sorted(os.listdir(self.root_dir)):
            ver_img_cnt = 0
            for i in range(ver_img_per_class):
                list_face = sorted(os.listdir(os.path.join(self.root_dir, iden, 'face')))
                list_ocular = sorted(os.listdir(os.path.join(self.root_dir, iden, 'periocular')))
                list_len = len(list_ocular)
                offset = list_len // ver_img_per_class
                self.face_img_dir_list.append(os.path.join(self.root_dir, iden, 'face', list_ocular[offset*i]))
                self.ocular_img_dir_list.append(os.path.join(self.root_dir, iden, 'periocular', list_ocular[offset*i]))
                self.label_list.append(cnt)
                ver_img_cnt += 1
                if ver_img_cnt == ver_img_per_class:
                    break
            cnt += 1

        self.onehot_label = np.zeros((len(self.ocular_img_dir_list), self.num_classes))
        for i in range(len(self.ocular_img_dir_list)):
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
        return len(self.ocular_img_dir_list)

    def __getitem__(self, idx):
        face = Image.open(self.face_img_dir_list[idx])
        face = self.face_transform(face)
        ocular = Image.open(self.ocular_img_dir_list[idx])
        ocular = self.ocular_transform(ocular)
        label = self.label_list[idx]
        onehot = self.onehot_label[idx]
        return face, ocular, label, onehot


