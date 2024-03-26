from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch

from .pooling import GeneralizedMeanPoolingP, GeneralizedMeanPooling
from .transformer import Transformer, forward_transformer, build_position_encoding

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False, norm=False, num_classes=0, pool_type='avgpool', ):
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNet.__factory[depth](pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)

        #self.base = nn.Sequential(
        #    resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        #    resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.initial_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        if pool_type == 'avgpool':
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        if pool_type == 'gempool':
            self.global_pool = GeneralizedMeanPooling(norm=3)

        self.norm = norm
        self.num_classes = num_classes

        # Append new layers
        # Change the num_features to CNN output channels
        out_planes = resnet.fc.in_features
        self.num_features = out_planes
        self.feat_bn = nn.BatchNorm1d(self.num_features)
        self.feat_bn.bias.requires_grad_(False)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        if self.num_classes > 0:
            self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
            init.normal_(self.classifier.weight, std=0.001)

        # transformer related modules
        self.feat_bn_2d = nn.BatchNorm2d(out_planes)
        self.feat_bn_2d.bias.requires_grad_(False)
        self.feat_bn_2d.apply(weights_init_kaiming)
        
        num_patch = 12
        tran_hidden_dim = 256
        self.positional_encoder = build_position_encoding(hidden_dim=tran_hidden_dim, pos_embed_type='learned')
        self.transformer = Transformer(d_model=tran_hidden_dim, dropout=0.1, nhead=8, dim_feedforward=2048, num_encoder_layers=6,
                                       num_decoder_layers=6, normalize_before=True, return_intermediate_dec=False)
        self.class_embed = nn.Linear(tran_hidden_dim, num_patch)
        self.query_embed = nn.Embedding(num_patch, tran_hidden_dim)
        self.input_proj = nn.Conv2d(2048, tran_hidden_dim, kernel_size=1)

        self.transformer_bottleneck = nn.BatchNorm1d(xxx)
        self.transformer_bottleneck.requires_grad_(False)
        self.transformer_bottleneck.apply(weights_init_kaiming)

        # parameter initialization
        if not pretrained:
            self.reset_params()

    def forward(self, x, lowlayer=False):
        bs, _, h, w = x.shape

        # x = self.base(x)
        x = self.initial_conv(x)
        x = self.layer1(x)

        if (lowlayer and (not self.training)):
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
            return x

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # compute bn on feature map
        featmap = self.feat_bn_2d(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        bn_x = self.feat_bn(x)  # bottleneck

        if (self.training is False):
            bn_x = F.normalize(bn_x)
            return bn_x

        if self.norm:
            bn_x = F.normalize(bn_x)

        # transformer forward during training
        mask = torch.zeros((bs, h, w), dtype=torch.bool, device=featmap.device)
        patch_features, outp_class = forward_transformer(
            mask, featmap, self.positional_encoder, self.transformer,
            self.input_proj, self.query_embed, self.class_embed)

        patch_features = self.transformer_bottleneck(patch_features)
        patch_features = F.normalize(patch_features)

        if self.num_classes > 0:
            prob = self.classifier(bn_x)
            return bn_x, prob
        else:
            return bn_x, x, patch_features


    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        resnet = ResNet.__factory[self.depth](pretrained=self.pretrained)
        self.base[0].load_state_dict(resnet.conv1.state_dict())
        self.base[1].load_state_dict(resnet.bn1.state_dict())
        self.base[2].load_state_dict(resnet.maxpool.state_dict())
        self.base[3].load_state_dict(resnet.layer1.state_dict())
        self.base[4].load_state_dict(resnet.layer2.state_dict())
        self.base[5].load_state_dict(resnet.layer3.state_dict())
        self.base[6].load_state_dict(resnet.layer4.state_dict())


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


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


