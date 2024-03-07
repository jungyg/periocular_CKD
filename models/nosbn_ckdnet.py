import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.utils.weight_norm as weight_norm
import torch.nn.functional as F
import math
from .blocks import *



class NoSBN_CKDNet(nn.Module):
    def __init__(self, block=IRBlock, layers=[2, 2, 2, 2], num_classes=1054):
        self.num_classes = num_classes
        self.inplanes = 64
        super(NoSBN_CKDNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        self.face_layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.inplanes = 256
        self.ocular_layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.face_bn4 = nn.BatchNorm2d(512)
        self.face_fc5 = nn.Linear(512 * 8 * 8, 512)
        self.face_bn5 = nn.BatchNorm1d(512)
        self.face_fc = nn.Linear(512, self.num_classes)

        self.ocular_bn4 = nn.BatchNorm2d(512)
        self.ocular_fc5 = nn.Linear(512 * 3 * 8, 512)
        self.ocular_bn5 = nn.BatchNorm1d(512)
        self.ocular_fc = nn.Linear(512, self.num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)



    def forward(self, face, ocular):

        face = self.conv1(face)
        face = self.bn1(face)
        face = self.relu(face)
        face = self.layer1(face)
        face = self.layer2(face)
        face = self.layer3(face)
        face = self.face_layer4(face)
        face = self.face_bn4(face)
        face = face.view(face.size(0), -1)
        face = self.face_fc5(face)
        face_feature = self.face_bn5(face)
        face_out = self.face_fc(face_feature)

        ocular = self.conv1(ocular)
        ocular = self.bn1(ocular)
        ocular = self.relu(ocular)
        ocular = self.layer1(ocular)
        ocular = self.layer2(ocular)
        ocular = self.layer3(ocular)
        ocular = self.ocular_layer4(ocular)
        ocular = self.ocular_bn4(ocular)
        ocular = ocular.view(ocular.size(0), -1)
        ocular = self.ocular_fc5(ocular)
        ocular_feature = self.ocular_bn5(ocular)
        ocular_out = self.ocular_fc(ocular_feature)

        return face_feature, ocular_feature, face_out, ocular_out

    def face_forward(self, face):
        face = self.conv1(face)
        face = self.bn1(face)
        face = self.relu(face)
        face = self.layer1(face)
        face = self.layer2(face)
        face = self.layer3(face)
        face = self.face_layer4(face)
        face = self.face_bn4(face)
        face = face.view(face.size(0), -1)
        face = self.face_fc5(face)
        feature = self.face_bn5(face)
        out = self.face_fc(feature)
        return feature, out

    def ocular_forward(self, ocular):
        ocular = self.conv1(ocular)
        ocular = self.bn1(ocular)
        ocular = self.relu(ocular)
        ocular = self.layer1(ocular)
        ocular = self.layer2(ocular)
        ocular = self.layer3(ocular)
        ocular = self.ocular_layer4(ocular)
        ocular = self.ocular_bn4(ocular)
        ocular = ocular.view(ocular.size(0), -1)
        ocular = self.ocular_fc5(ocular)
        feature = self.ocular_bn5(ocular)
        out = self.ocular_fc(feature)
        return feature, out

    def get_ocular_feature(self, ocular):
        ocular = self.conv1(ocular)
        ocular = self.bn1(ocular)
        ocular = self.relu(ocular)
        ocular = self.layer1(ocular)
        ocular = self.layer2(ocular)
        ocular = self.layer3(ocular)
        ocular = self.ocular_layer4(ocular)
        ocular = self.ocular_bn4(ocular)
        ocular = ocular.view(ocular.size(0), -1)
        ocular = self.ocular_fc5(ocular)
        feature = self.ocular_bn5(ocular)
        return feature

    def get_face_feature(self, face):
        face = self.conv1(face)
        face = self.bn1(face)
        face = self.relu(face)
        face = self.layer1(face)
        face = self.layer2(face)
        face = self.layer3(face)
        face = self.face_layer4(face)
        face = self.face_bn4(face)
        face = face.view(face.size(0), -1)
        face = self.face_fc5(face)
        feature = self.face_bn5(face)
        return feature

    def get_ocular_logit(self, ocular):
        ocular = self.conv1(ocular)
        ocular = self.bn1(ocular)
        ocular = self.relu(ocular)
        ocular = self.layer1(ocular)
        ocular = self.layer2(ocular)
        ocular = self.layer3(ocular)
        ocular = self.ocular_layer4(ocular)
        ocular = self.ocular_bn4(ocular)
        ocular = ocular.view(ocular.size(0), -1)
        ocular = self.ocular_fc5(ocular)
        feature = self.ocular_bn5(ocular)
        out = self.ocular_fc(feature)
        return out

    def get_face_logit(self, face):
        face = self.conv1(face)
        face = self.bn1(face)
        face = self.relu(face)
        face = self.layer1(face)
        face = self.layer2(face)
        face = self.layer3(face)
        face = self.face_layer4(face)
        face = self.face_bn4(face)
        face = face.view(face.size(0), -1)
        face = self.face_fc5(face)
        feature = self.face_bn5(face)
        out = self.face_fc(feature)
        return out


