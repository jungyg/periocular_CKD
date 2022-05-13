import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.utils.weight_norm as weight_norm
import torch.nn.functional as F
import math
from .blocks import *
from .face_network import face_network
from .ocular_network import ocular_network


class separate_face_ocular_net(nn.Module):
    def __init__(self, block=shared_block, layers=[2, 2, 2, 2], num_classes=1054):
        super(separate_face_ocular_net, self).__init__()
        self.face_network = face_network()
        self.ocular_network = ocular_network()


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def forward(self, face, ocular):
        face_feature, face_out = self.face_network(face)
        ocular_feature, ocular_out = self.ocular_network(ocular)
        return face_feature, ocular_feature, face_out, ocular_out

    def face_forward(self, face):
        feature, out = self.face_network(face)
        return feature, out

    def ocular_forward(self, ocular):
        feature, out = self.ocular_network(ocular)
        return feature, out

    def get_ocular_feature(self, ocular):
        feature = self.ocular_network.get_ocular_feature(ocular)
        return feature

    def get_face_feature(self, face):
        feature = self.face_network.get_face_feature(face)
        return feature

    def get_ocular_logit(self, ocular):
        out = self.ocular_network.get_ocular_logit(ocular)
        return out

    def get_face_logit(self, face):
        out = self.face_network.get_face_logit(face)
        return out

