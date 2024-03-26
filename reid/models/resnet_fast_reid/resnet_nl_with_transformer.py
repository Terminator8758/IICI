# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging
import math

import torch
from torch import nn
from torch.nn import functional as F

from .non_local import Non_local
from .batch_norm import BatchNorm, IBN, get_norm
from .pooling import GeneralizedMeanPoolingP, GeneralizedMeanPooling
from ..transformer import Transformer, forward_transformer, build_position_encoding
from spcl.models.mixstyle import MixStyle

__all__ = ['resnet50', 'ResNet']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, bn_norm, with_ibn=False, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if with_ibn:
            self.bn1 = IBN(planes, bn_norm)
        else:
            self.bn1 = get_norm(bn_norm, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = get_norm(bn_norm, planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, bn_norm, with_ibn=False, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if with_ibn:
            self.bn1 = IBN(planes, bn_norm)
        else:
            self.bn1 = get_norm(bn_norm, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = get_norm(bn_norm, planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = get_norm(bn_norm, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Detr_transformer(nn.Module):
    def __init__(self, num_patch=12, tran_hidden_dim=256):
        super().__init__()
        self.positional_encoder = build_position_encoding(hidden_dim=tran_hidden_dim, pos_embed_type='learned')
        self.transformer = Transformer(d_model=tran_hidden_dim, dropout=0.1, nhead=8, dim_feedforward=2048,
                                       num_encoder_layers=6,
                                       num_decoder_layers=6, normalize_before=True, return_intermediate_dec=False)
        self.class_embed = nn.Linear(tran_hidden_dim, num_patch)
        self.query_embed = nn.Embedding(num_patch, tran_hidden_dim)
        self.input_proj = nn.Conv2d(2048, tran_hidden_dim, kernel_size=1)

        self.h = 256
        self.w = 128
        #self.trans_bottleneck = nn.BatchNorm1d(3072)  # 256*12=3072
        #self.trans_bottleneck.bias.requires_grad_(False)
        #self.trans_bottleneck.apply(weights_init_kaiming)

    def forward(self, featmap):
        mask = torch.zeros((len(featmap), self.h, self.w), dtype=torch.bool, device=featmap.device)
        patch_features, outp_class = forward_transformer(
            mask, featmap, self.positional_encoder, self.transformer,
            self.input_proj, self.query_embed, self.class_embed)  # patch_features shape: [bs, 12, 256]

        tran_feat = patch_features.reshape((patch_features.size(0), -1))
        #tran_feat = self.trans_bottleneck(patch_feature_concated)
        tran_feat = F.normalize(tran_feat)
        #tran_feat = torch.chunk(tran_feat, 2, dim=1)  # split the patch features into three parts
        return tran_feat



class ResNet(nn.Module):
    """
        ResNet-50 class with Non-local option, IBN option, and GeM pooling enabled.
    """
    def __init__(self, last_stride, bn_norm, with_ibn, with_nl, block, layers, non_layers, pool_type='gempool', has_mixstyle=False):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = get_norm(bn_norm, 64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0], 1, bn_norm, with_ibn)
        self.layer2 = self._make_layer(block, 128, layers[1], 2, bn_norm, with_ibn)
        self.layer3 = self._make_layer(block, 256, layers[2], 2, bn_norm, with_ibn)
        self.layer4 = self._make_layer(block, 512, layers[3], last_stride, bn_norm)

        # added head layers:
        if pool_type == 'avgpool':     self.global_pool = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'gempoolP':    self.global_pool = GeneralizedMeanPoolingP()
        elif pool_type == 'gempool':     self.global_pool = GeneralizedMeanPooling()  # default choice
        else:                            raise KeyError(f"{pool_type} is not supported!")

        #self.feat_bn = nn.BatchNorm1d(2048)  # bottleneck layer
        #self.feat_bn.bias.requires_grad_(False)

        self.random_init()

        if with_nl: self._build_nonlocal(layers, non_layers, bn_norm)
        else:       self.NL_1_idx = self.NL_2_idx = self.NL_3_idx = self.NL_4_idx = []

        self.has_mixstyle = has_mixstyle
        if has_mixstyle:
            mixstyle_p = 0.5
            mixstyle_alpha = 0.1
            self.mixstyle = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha, mix='random')

        # transformer related modules
        self.feat_bn_2d = nn.BatchNorm2d(2048)
        self.feat_bn_2d.bias.requires_grad_(False)
        self.feat_bn_2d.apply(weights_init_kaiming)
        
        num_patch = 12
        tran_hidden_dim = 256
        self.positional_encoder = build_position_encoding(hidden_dim=tran_hidden_dim, pos_embed_type='learned')
        self.transformer = Transformer(d_model=tran_hidden_dim, dropout=0.1, nhead=8, dim_feedforward=2048,
                                       num_encoder_layers=6,
                                       num_decoder_layers=6, normalize_before=True, return_intermediate_dec=False)
        self.class_embed = nn.Linear(tran_hidden_dim, num_patch)
        self.query_embed = nn.Embedding(num_patch, tran_hidden_dim)
        self.input_proj = nn.Conv2d(2048, tran_hidden_dim, kernel_size=1)

        self.trans_bottleneck = nn.BatchNorm1d(3072)  # 256*12=3072
        self.trans_bottleneck.bias.requires_grad_(False)
        self.trans_bottleneck.apply(weights_init_kaiming)


    def _make_layer(self, block, planes, blocks, stride=1, bn_norm="BN", with_ibn=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                get_norm(bn_norm, planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, bn_norm, with_ibn, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_norm, with_ibn))

        return nn.Sequential(*layers)


    def _build_nonlocal(self, layers, non_layers, bn_norm):
        self.NL_1 = nn.ModuleList(
            [Non_local(256, bn_norm) for _ in range(non_layers[0])])
        self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
        self.NL_2 = nn.ModuleList(
            [Non_local(512, bn_norm) for _ in range(non_layers[1])])
        self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
        self.NL_3 = nn.ModuleList(
            [Non_local(1024, bn_norm) for _ in range(non_layers[2])])
        self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
        self.NL_4 = nn.ModuleList(
            [Non_local(2048, bn_norm) for _ in range(non_layers[3])])
        self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])


    def forward(self, x, lowlayer=False):
        bs, _, h, w = x.shape

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # layer 1
        NL1_counter = 0
        if len(self.NL_1_idx) == 0:
            self.NL_1_idx = [-1]
        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
            if i == self.NL_1_idx[NL1_counter]:
                _, C, H, W = x.shape
                x = self.NL_1[NL1_counter](x)
                NL1_counter += 1

        if self.has_mixstyle:
            x = self.mixstyle(x)

        # layer 2
        NL2_counter = 0
        if len(self.NL_2_idx) == 0:
            self.NL_2_idx = [-1]
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
            if i == self.NL_2_idx[NL2_counter]:
                _, C, H, W = x.shape
                x = self.NL_2[NL2_counter](x)
                NL2_counter += 1

        if self.has_mixstyle:
            x = self.mixstyle(x)

        if ((not self.training) and lowlayer):  # new added for intra-cameea ID split
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
            return x

        # layer 3
        NL3_counter = 0
        if len(self.NL_3_idx) == 0:
            self.NL_3_idx = [-1]
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)
            if i == self.NL_3_idx[NL3_counter]:
                _, C, H, W = x.shape
                x = self.NL_3[NL3_counter](x)
                NL3_counter += 1

        # layer 4
        NL4_counter = 0
        if len(self.NL_4_idx) == 0:
            self.NL_4_idx = [-1]
        for i in range(len(self.layer4)):
            x = self.layer4[i](x)
            if i == self.NL_4_idx[NL4_counter]:
                _, C, H, W = x.shape
                x = self.NL_4[NL4_counter](x)
                NL4_counter += 1

        # compute bn on feature map
        featmap = self.feat_bn_2d(x)

        x = self.global_pool(x)
        #x = x.view(x.size(0), -1)
        #bn_x = self.feat_bn(x)
        bn_x = self.feat_bn_2d(x)
        bn_x = bn_x[..., 0, 0]  

        bn_x = F.normalize(bn_x)

        if (not self.training):
            return bn_x

        # transformer forward during training
        #print('batch featmap shape= ', featmap.shape)
        mask = torch.zeros((bs, h, w), dtype=torch.bool, device=featmap.device)
        patch_features, outp_class = forward_transformer(
            mask, featmap, self.positional_encoder, self.transformer,
            self.input_proj, self.query_embed, self.class_embed)  # patch_features shape: [bs, 12, 256]

        tran_feat = patch_features.reshape((patch_features.size(0), -1))
        #tran_feat = self.trans_bottleneck(patch_feature_concated)
        tran_feat = F.normalize(tran_feat)
        #tran_feat = torch.chunk(tran_feat, 2, dim=1)  # split the patch features into three parts
        return bn_x, tran_feat


    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def resnet50(**kwargs):
    model = ResNet(last_stride=1, bn_norm='BN', with_ibn=False, block=Bottleneck,
                   layers=[3, 4, 6, 3], non_layers=[0, 2, 3, 0], **kwargs)  # kwargs: with_nl=True, pool_type='gem'
    pretrained_path = 'reid/models/resnet_fast_reid/resnet50-19c8e357.pth'
    state_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)
    return model

