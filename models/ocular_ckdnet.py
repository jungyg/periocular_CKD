import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.utils.weight_norm as weight_norm
import torch.nn.functional as F
import math
from .blocks import *



class CKDNetwork_ocular(nn.Module):
    def __init__(self, block=shared_block, layers=[2, 2, 2, 2], num_classes=1054):
        self.num_classes = num_classes
        self.inplanes = 64
        super(CKDNetwork_ocular, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.block1 = self._make_layer(IRBlock, 64, 2, stride=2)
        self.block2 = self._make_layer(IRBlock, 128, 2, stride=2)
        self.block3 = self._make_layer(IRBlock, 256, 2, stride=2)
        self.ocular1_layer4 = self._make_layer(IRBlock, 512, 2, stride=2)
        self.inplanes = 256
        self.ocular2_layer4 = self._make_layer(IRBlock, 512, 2, stride=2)

        self.ocular1_bn4 = nn.BatchNorm2d(512)
        self.ocular2_bn4 = nn.BatchNorm2d(512)

        self.ocular1_fc5 = nn.Linear(512 * 3 * 8, 512)
        self.ocular1_bn5 = nn.BatchNorm1d(512)
        self.ocular1_fc = nn.Linear(512, self.num_classes)

        self.ocular2_fc5 = nn.Linear(512 * 3 * 8, 512)
        self.ocular2_bn5 = nn.BatchNorm1d(512)
        self.ocular2_fc = nn.Linear(512, self.num_classes)

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

    def forward(self, ocular):
        ocular = self.conv1(ocular)
        ocular = self.bn1(ocular)
        ocular = self.relu(ocular)

        ocular = self.block1(ocular)
        ocular = self.block2(ocular)
        ocular = self.block3(ocular)

        ocular1 = self.ocular1_layer4(ocular)
        ocular2 = self.ocular2_layer4(ocular)

        ocular1 = self.ocular1_bn4(ocular1)
        ocular1 = ocular1.view(ocular1.size(0), -1)
        ocular1 = self.ocular1_fc5(ocular1)
        ocular1_feature = self.ocular1_bn5(ocular1)
        ocular1_out = self.ocular1_fc(ocular1_feature)

        ocular2 = self.ocular2_bn4(ocular2)
        ocular2 = ocular2.view(ocular2.size(0), -1)
        ocular2 = self.ocular2_fc5(ocular2)
        ocular2_feature = self.ocular2_bn5(ocular2)
        ocular2_out = self.ocular2_fc(ocular2_feature)

        return ocular1_feature, ocular2_feature, ocular1_out, ocular2_out


    def ocular_forward(self, ocular):
        ocular = self.conv1(ocular)
        ocular = self.bn1(ocular)
        ocular = self.relu(ocular)

        ocular = self.block1(ocular)
        ocular = self.block2(ocular)
        ocular = self.block3(ocular)
        ocular = self.ocular1_layer4(ocular)
        ocular = self.ocular1_bn4(ocular)
        ocular = ocular.view(ocular.size(0), -1)

        ocular = self.ocular1_fc5(ocular)
        feature = self.ocular1_bn5(ocular)
        out = self.ocular1_fc(feature)
        return feature, out

    def get_ocular_feature(self, ocular):
        ocular = self.conv1(ocular)
        ocular = self.bn1(ocular)
        ocular = self.relu(ocular)

        ocular = self.block1(ocular)
        ocular = self.block2(ocular)
        ocular = self.block3(ocular)
        ocular = self.ocular1_layer4(ocular)
        ocular = self.ocular1_bn4(ocular)
        ocular = ocular.view(ocular.size(0), -1)

        ocular = self.ocular1_fc5(ocular)
        feature = self.ocular1_bn5(ocular)
        return feature


    def get_ocular_logit(self, ocular):
        ocular = self.conv1(ocular)
        ocular = self.bn1(ocular)
        ocular = self.relu(ocular)

        ocular = self.block1(ocular)
        ocular = self.block2(ocular)
        ocular = self.block3(ocular)
        ocular = self.ocular1_layer4(ocular)
        ocular = self.ocular1_bn4(ocular)
        ocular = ocular.view(ocular.size(0), -1)


        ocular = self.ocular1_fc5(ocular)
        ocular = self.ocular1_bn5(ocular)
        out = self.ocular1_fc(ocular)
        return out

