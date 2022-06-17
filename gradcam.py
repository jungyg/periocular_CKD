import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
import shared_networks2 as shared_networks
import networks
import torch.utils.data as data
from PIL import Image
import os
from collections import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm


class train_dataset(data.Dataset):
    def __init__(self, dset_type):
        dset = 'trainingdb'
        assert dset_type in ['train', 'val', 'test']
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
        raw_image = cv2.imread(self.ocular_img_dir_list[idx])
        raw_image = cv2.resize(raw_image, (128,48))
        raw = Image.open(self.ocular_img_dir_list[idx])
        ocular = self.ocular_transform(raw)
        label = self.label_list[idx]
        onehot = self.onehot_label[idx]
        return raw_image, ocular, label, onehot



def preprocess(image_path):
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image


class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        self.image_shape = image.shape[2:]
        self.logits = self.model(image)[1]
        self.probs = F.softmax(self.logits, dim=1)
        return self.probs.sort(dim=1, descending=True)  # ordered results

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


class BackPropagation(_BaseWrapper):
    def forward(self, image):
        self.image = image.requires_grad_()
        return super(BackPropagation, self).forward(self.image)

    def generate(self):
        gradient = self.image.grad.clone()
        self.image.grad.zero_()
        return gradient


class GuidedBackPropagation(BackPropagation):
    """
    "Striving for Simplicity: the All Convolutional Net"
    https://arxiv.org/pdf/1412.6806.pdf
    Look at Figure 1 on page 8.
    """

    def __init__(self, model):
        super(GuidedBackPropagation, self).__init__(model)

        def backward_hook(module, grad_in, grad_out):
            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (F.relu(grad_in[0]),)

        for module in self.model.named_modules():
            self.handlers.append(module[1].register_backward_hook(backward_hook))


class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list

        def save_fmaps(key):
            def forward_hook(module, input, output):
                if len(output) != 1:
                    output = output[1]
                self.fmap_pool[key] = output.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam


def occlusion_sensitivity(
    model, images, ids, mean=None, patch=35, stride=1, n_batches=128
):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure A5 on page 17
    Originally proposed in:
    "Visualizing and Understanding Convolutional Networks"
    https://arxiv.org/abs/1311.2901
    """

    torch.set_grad_enabled(False)
    model.eval()
    mean = mean if mean else 0
    patch_H, patch_W = patch if isinstance(patch, Sequence) else (patch, patch)
    pad_H, pad_W = patch_H // 2, patch_W // 2

    # Padded image
    images = F.pad(images, (pad_W, pad_W, pad_H, pad_H), value=mean)
    B, _, H, W = images.shape
    new_H = (H - patch_H) // stride + 1
    new_W = (W - patch_W) // stride + 1

    # Prepare sampling grids
    anchors = []
    grid_h = 0
    while grid_h <= H - patch_H:
        grid_w = 0
        while grid_w <= W - patch_W:
            grid_w += stride
            anchors.append((grid_h, grid_w))
        grid_h += stride

    # Baseline score without occlusion
    baseline = model(images).detach().gather(1, ids)

    # Compute per-pixel logits
    scoremaps = []
    for i in tqdm(range(0, len(anchors), n_batches), leave=False):
        batch_images = []
        batch_ids = []
        for grid_h, grid_w in anchors[i : i + n_batches]:
            images_ = images.clone()
            images_[..., grid_h : grid_h + patch_H, grid_w : grid_w + patch_W] = mean
            batch_images.append(images_)
            batch_ids.append(ids)
        batch_images = torch.cat(batch_images, dim=0)
        batch_ids = torch.cat(batch_ids, dim=0)
        scores = model(batch_images).detach().gather(1, batch_ids)
        scoremaps += list(torch.split(scores, B))

    diffmaps = torch.cat(scoremaps, dim=1) - baseline
    diffmaps = diffmaps.view(B, new_H, new_W)

    return diffmaps


import copy
import os.path as osp

import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms



def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def load_images(image_paths):
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    return images, raw_images


def get_classtable():
    classes = []
    with open("samples/synset_words.txt") as lines:
        for line in lines:
            line = line.strip().split(" ", 1)[1]
            line = line.split(", ", 1)[0].replace(" ", "_")
            classes.append(line)
    return classes



def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))
    raw_name = filename.split('.')[0] + '_raw' + '.jpg'
    cv2.imwrite(raw_name, raw_image)


def save_sensitivity(filename, maps):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename, maps)




def main(shared=True, topk=1, output_dir='./', cuda="0", idx=0, train='train'):
    """
    Visualize model responses given multiple images
    """
    if train == 'train':
        dset = train_dataset(dset_type='train')
    else:
        dset = train_dataset(dset_type='val')


    device = get_device(cuda)


    # Model from torchvision

    if shared:
        model = shared_networks.shared_network(num_classes=1054)
        model.load_state_dict(torch.load('./results/exp_01/model.pth.tar', map_location='cpu')['state_dict'])
        model.to(device)
        model.eval()
        target_layer = 'ocular_bn4'

    else:
        model = networks.O_net(num_classes=1054)
        model.load_state_dict(torch.load('./results/exp_06/model.pth.tar', map_location='cpu')['state_dict'])
        model.to(device)
        model.eval()
        target_layer = 'bn4'


    # Images
    raw_image, images, label, _ = dset[idx]
    images = images.unsqueeze(0).to(device)
    # images = torch.stack(images).to(device)

    """
    Common usage:
    1. Wrap your model with visualization classes defined in grad_cam.py
    2. Run forward() with images
    3. Run backward() with a list of specific classes
    4. Run generate() to export results
    """

    # =========================================================================
    bp = BackPropagation(model=model)
    probs, ids = bp.forward(images)  # sorted



    # =========================================================================
    print("Grad-CAM")

    gcam = GradCAM(model=model)
    _ = gcam.forward(images)

    gbp = GuidedBackPropagation(model=model)
    _ = gbp.forward(images)

    for i in range(topk):
        # Guided Backpropagation
        gbp.backward(ids=ids[:, [i]])
        gradients = gbp.generate()

        # Grad-CAM
        gcam.backward(ids=ids[:, [i]])
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            print("\t#{}: ({:.5f})".format(j,  probs[j, i]))

            # Guided Backpropagation
            # Guided Backpropagation
            # save_gradient(
            #     filename=osp.join(
            #         output_dir,
            #         "guided-{}-{}-{}.png".format(
            #             target_layer, shared, train
            #         ),
            #     ),
            #     gradient=gradients[j],
            # )

            # Grad-CAM
            save_gradcam(
                filename=osp.join(
                    output_dir,
                    "0-{}-{}-gradcam-{}.png".format(
                         train, shared, target_layer
                    ),
                ),
                gcam=regions[j, 0],raw_image = raw_image
            )



            # Guided Grad-CAM
            save_gradient(
                filename=osp.join(
                    output_dir,
                    "0-{}-{}-guided_gradcam-{}.png".format(
                        train, shared, target_layer
                    ),
                ),
                gradient=torch.mul(regions, gradients)[j],
            )




if __name__ == "__main__":
    idx = np.random.randint(25000)
    main(shared=False, topk=1, output_dir='', cuda="0", train='train', idx=idx)
    main(shared=True, topk=1, output_dir='', cuda="0", train='train', idx=idx)
    main(shared=False, topk=1, output_dir='', cuda="0", train='val', idx=idx)
    main(shared=True, topk=1, output_dir='', cuda="0", train='val', idx=idx)