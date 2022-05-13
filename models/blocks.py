import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.utils.weight_norm as weight_norm
import torch.nn.functional as F
import math


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

def unify_and_reshape(face, ocular):
    fn, fc, fh, fw = face.size()
    on, oc, oh, ow = ocular.size()
    face = face.view(fn, fc, 1, fh*fw)
    ocular = ocular.view(on, oc, 1, oh*ow)
    out = torch.cat((face, ocular), dim=3)
    offset = fh*fw
    return offset, out

def split_and_reshape(tensor, offset, face, ocular):
    fn, fc, fh, fw = face.shape
    on, oc, oh, ow = ocular.shape
    face_out = tensor[:, :, :, :offset].view(fn, fc, fh, fw)
    ocular_out = tensor[:, :, :, offset:].view(on, oc, oh, ow)
    return face_out, ocular_out

class shared_block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, use_se=True):
        super(shared_block, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.act1 = nn.ReLU()
        self.conv1 = conv3x3(in_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.act2 = nn.ReLU()
        self.conv2 = conv3x3(out_planes, out_planes, stride)
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(out_planes)

        if self.stride != 1:
            self.downsample = downsample_layer(in_planes, out_planes, stride)
        else:
            self.downsample = None

    def forward(self, face, ocular):
        face_residual = face
        ocular_residual = ocular
        offset, reshaped = unify_and_reshape(face, ocular)
        reshaped = self.bn1(reshaped)
        reshaped = self.act1(reshaped)
        face, ocular = split_and_reshape(reshaped, offset, face, ocular)
        face = self.conv1(face)
        ocular = self.conv1(ocular)
        offset, reshaped = unify_and_reshape(face, ocular)
        reshaped = self.bn2(reshaped)
        reshaped = self.act2(reshaped)
        face, ocular = split_and_reshape(reshaped, offset, face, ocular)
        face_out = self.conv2(face)
        ocular_out = self.conv2(ocular)
        if self.use_se:
            face_out = self.se(face_out)
            ocular_out = self.se(ocular_out)

        if self.downsample is not None:
            face_residual, ocular_residual = self.downsample(face_residual, ocular_residual)

        face_out += face_residual
        ocular_out += ocular_residual

        return face_out, ocular_out

    def face_forward(self, face):
        residual = face
        face = self.bn1(face)
        face = self.act1(face)
        face = self.conv1(face)
        face = self.bn2(face)
        face = self.act2(face)
        face_out = self.conv2(face)
        if self.use_se:
            face_out = self.se(face_out)

        if self.downsample is not None:
            residual = self.downsample.face_forward(residual)

        face_out += residual

        return face_out

    def ocular_forward(self, ocular):
        residual = ocular
        ocular = self.bn1(ocular)
        ocular = self.act1(ocular)
        ocular = self.conv1(ocular)
        ocular = self.bn2(ocular)
        ocular = self.act2(ocular)
        ocular_out = self.conv2(ocular)
        if self.use_se:
            ocular_out = self.se(ocular_out)

        if self.downsample is not None:
            residual = self.downsample.ocular_forward(residual)

        ocular_out += residual

        return ocular_out

class IRBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.act1 = nn.ReLU()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, stride)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.act1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

class downsample_layer(nn.Module):
    def __init__(self, in_planes, out_planes, stride):
        super(downsample_layer, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, face, ocular):
        face = self.conv(face)
        ocular = self.conv(ocular)
        offset, reshaped = unify_and_reshape(face, ocular)
        reshaped = self.bn(reshaped)
        face, ocular = split_and_reshape(reshaped, offset, face, ocular)
        return face, ocular

    def face_forward(self, face):
        face = self.conv(face)
        face = self.bn(face)
        return face

    def ocular_forward(self, ocular):
        ocular = self.conv(ocular)
        ocular = self.bn(ocular)
        return ocular


