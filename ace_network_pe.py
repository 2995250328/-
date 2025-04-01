# Copyright © Niantic, Inc. 2022.

import logging
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from wtconv import WTConv2d

_logger = logging.getLogger(__name__)

'''-------------一、SE模块-----------------------------'''
#全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid
class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )
 
    def forward(self, x):
            # 读取批数据图片数量及通道数
            b, c, h, w = x.size()
            # Fsq操作：经池化后输出b*c的矩阵
            y = self.gap(x).view(b, c)
            # Fex操作：经全连接层输出（b，c，1，1）矩阵
            y = self.fc(y).view(b, c, 1, 1)
            # Fscale操作：将得到的权重乘以原来的特征图x
            return x * y.expand_as(x)

class PyramidPooling(nn.Module):
    def __init__(self, in_channels, up_kwargs=None):
        super(PyramidPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

        out_channels = in_channels // 4  # 512 / 4 = 128

        self.conv = nn.Conv2d(1,in_channels,1,bias=False)

        self.conv1 = nn.Sequential(
            WTConv2d(in_channels, in_channels, 1, bias=False,wt_levels=2),
            nn.Conv2d(in_channels,out_channels,1,bias=False),
            nn.LayerNorm([out_channels, 1, 1]),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            WTConv2d(in_channels, in_channels, 1, bias=False,wt_levels=2),
            nn.Conv2d(in_channels,out_channels,1,bias=False),
            nn.LayerNorm([out_channels, 2, 2]),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            WTConv2d(in_channels, in_channels, 1, bias=False,wt_levels=2),
            nn.Conv2d(in_channels,out_channels,1,bias=False),
            nn.LayerNorm([out_channels, 3, 3]),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            WTConv2d(in_channels, in_channels, 1, bias=False,wt_levels=2),
            nn.Conv2d(in_channels,out_channels,1,bias=False),
            nn.LayerNorm([out_channels, 6, 6]),
            nn.ReLU(inplace=True)
        )

        self._up_kwargs = up_kwargs if up_kwargs else {"mode": "bilinear", "align_corners": False}

    def forward(self, x):       
        x = self.conv(x)
        _, _, h, w = x.size()
        h_enc = math.ceil(h / 8)
        w_enc = math.ceil(w / 8)
        feat1 = F.interpolate(self.conv1(self.pool1(x)), size=(h_enc, w_enc), **self._up_kwargs)
        feat2 = F.interpolate(self.conv2(self.pool2(x)), size=(h_enc, w_enc), **self._up_kwargs)
        feat3 = F.interpolate(self.conv3(self.pool3(x)), size=(h_enc, w_enc), **self._up_kwargs)
        feat4 = F.interpolate(self.conv4(self.pool4(x)), size=(h_enc, w_enc), **self._up_kwargs)
        return torch.cat((feat1, feat2, feat3, feat4), 1)

class Encoder(nn.Module):
    """
    FCN encoder, used to extract features from the input images.

    The number of output channels is configurable, the default used in the paper is 512.
    """

    def __init__(self, out_channels=512):
        super(Encoder, self).__init__()

        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)

        self.res1_conv1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.res1_conv2 = nn.Conv2d(256, 256, 1, 1, 0)
        self.res1_conv3 = nn.Conv2d(256, 256, 3, 1, 1)

        self.res2_conv1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.res2_conv2 = nn.Conv2d(512, 512, 1, 1, 0)
        self.res2_conv3 = nn.Conv2d(512, self.out_channels, 3, 1, 1)

        self.res2_skip = nn.Conv2d(256, self.out_channels, 1, 1, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        res = F.relu(self.conv4(x))

        x = F.relu(self.res1_conv1(res))
        x = F.relu(self.res1_conv2(x))
        x = F.relu(self.res1_conv3(x))

        res = res + x

        x = F.relu(self.res2_conv1(res))
        x = F.relu(self.res2_conv2(x))
        x = F.relu(self.res2_conv3(x))

        x = self.res2_skip(res) + x

        return x


class Head(nn.Module):
    """
    MLP network predicting per-pixel scene coordinates given a feature vector. All layers are 1x1 convolutions.
    """

    def __init__(self,
                 mean,
                 num_head_blocks,
                 use_homogeneous,
                 homogeneous_min_scale=0.01,
                 homogeneous_max_scale=4.0,
                 in_channels=512):
        super(Head, self).__init__()

        self.use_homogeneous = use_homogeneous
        self.in_channels = in_channels  # Number of encoder features.
        self.head_channels = 512  # Hardcoded.

        # We may need a skip layer if the number of features output by the encoder is different.
        self.head_skip = nn.Identity() if self.in_channels == self.head_channels else nn.Conv2d(self.in_channels,
                                                                                                self.head_channels, 1,
                                                                                                1, 0)

        self.res3_conv1 = nn.Conv2d(self.in_channels, self.head_channels, 1, 1, 0)
        self.res3_conv2 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
        self.res3_conv3 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)

        self.res_blocks = []

        for block in range(num_head_blocks):
            self.res_blocks.append((
                nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
                nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
                nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
            ))

            super(Head, self).add_module(str(block) + 'c0', self.res_blocks[block][0])
            super(Head, self).add_module(str(block) + 'c1', self.res_blocks[block][1])
            super(Head, self).add_module(str(block) + 'c2', self.res_blocks[block][2])

        self.fc1 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
        self.fc2 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)

        if self.use_homogeneous:
            self.fc3 = nn.Conv2d(self.head_channels, 4, 1, 1, 0)

            # Use buffers because they need to be saved in the state dict.
            self.register_buffer("max_scale", torch.tensor([homogeneous_max_scale]))
            self.register_buffer("min_scale", torch.tensor([homogeneous_min_scale]))
            self.register_buffer("max_inv_scale", 1. / self.max_scale)
            self.register_buffer("h_beta", math.log(2) / (1. - self.max_inv_scale))
            self.register_buffer("min_inv_scale", 1. / self.min_scale)
        else:
            self.fc3 = nn.Conv2d(self.head_channels, 3, 1, 1, 0)

        # Learn scene coordinates relative to a mean coordinate (e.g. center of the scene).
        self.register_buffer("mean", mean.clone().detach().view(1, 3, 1, 1))

    def forward(self, res):

        x = F.relu(self.res3_conv1(res))
        x = F.relu(self.res3_conv2(x))
        x = F.relu(self.res3_conv3(x))

        res = self.head_skip(res) + x

        for res_block in self.res_blocks:
            x = F.relu(res_block[0](res))
            x = F.relu(res_block[1](x))
            x = F.relu(res_block[2](x))

            res = res + x

        sc = F.relu(self.fc1(res))
        sc = F.relu(self.fc2(sc))
        sc = self.fc3(sc)

        if self.use_homogeneous:
            # Dehomogenize coords:
            # Softplus ensures we have a smooth homogeneous parameter with a minimum value = self.max_inv_scale.
            h_slice = F.softplus(sc[:, 3, :, :].unsqueeze(1), beta=self.h_beta.item()) + self.max_inv_scale
            h_slice.clamp_(max=self.min_inv_scale)
            sc = sc[:, :3] / h_slice

        # Add the mean to the predicted coordinates.
        sc += self.mean

        return sc


class Regressor(nn.Module):
    """
    FCN architecture for scene coordinate regression.

    The network predicts a 3d scene coordinates, the output is subsampled by a factor of 8 compared to the input.
    """

    OUTPUT_SUBSAMPLE = 8

    def __init__(self, mean, num_head_blocks, use_homogeneous, num_encoder_features=512):
        """
        Constructor.

        mean: Learn scene coordinates relative to a mean coordinate (e.g. the center of the scene).
        num_head_blocks: How many extra residual blocks to use in the head (one is always used).
        use_homogeneous: Whether to learn homogeneous or 3D coordinates.
        num_encoder_features: Number of channels output of the encoder network.
        """
        super(Regressor, self).__init__()

        self.feature_dim = num_encoder_features

        self.encoder = Encoder(out_channels=self.feature_dim)
        self.se = SE_Block(inchannel=self.feature_dim)
        self.pyramidpooling = PyramidPooling(in_channels=self.feature_dim)
        #维度修改
        self.heads = Head(mean, num_head_blocks, use_homogeneous, in_channels=self.feature_dim*2)

    @classmethod
    def create_from_encoder(cls, encoder_state_dict, mean, num_head_blocks, use_homogeneous):
        """
        Create a regressor using a pretrained encoder, loading encoder-specific parameters from the state dict.

        encoder_state_dict: pretrained encoder state dictionary.
        mean: Learn scene coordinates relative to a mean coordinate (e.g. the center of the scene).
        num_head_blocks: How many extra residual blocks to use in the head (one is always used).
        use_homogeneous: Whether to learn homogeneous or 3D coordinates.
        """

        # Number of output channels of the last encoder layer.
        num_encoder_features = encoder_state_dict['res2_conv3.weight'].shape[0]

        # Create a regressor.
        _logger.info(f"Creating Regressor using pretrained encoder with {num_encoder_features} feature size.")
        regressor = cls(mean, num_head_blocks, use_homogeneous, num_encoder_features)

        # Load encoder weights.
        regressor.encoder.load_state_dict(encoder_state_dict)

        # Done.
        return regressor

    @classmethod
    def create_from_state_dict(cls, state_dict):
        """
        Instantiate a regressor from a pretrained state dictionary.

        state_dict: pretrained state dictionary.
        """
        # Mean is zero (will be loaded from the state dict).
        mean = torch.zeros((3,))

        # Count how many head blocks are in the dictionary.
        pattern = re.compile(r"^heads\.\d+c0\.weight$")
        num_head_blocks = sum(1 for k in state_dict.keys() if pattern.match(k))

        # Whether the network uses homogeneous coordinates.
        use_homogeneous = state_dict["heads.fc3.weight"].shape[0] == 4

        # Number of output channels of the last encoder layer.
        num_encoder_features = state_dict['encoder.res2_conv3.weight'].shape[0]

        # Create a regressor.
        _logger.info(f"Creating regressor from pretrained state_dict:"
                     f"\n\tNum head blocks: {num_head_blocks}"
                     f"\n\tHomogeneous coordinates: {use_homogeneous}"
                     f"\n\tEncoder feature size: {num_encoder_features}")
        regressor = cls(mean, num_head_blocks, use_homogeneous, num_encoder_features)

        # Load all weights.
        regressor.load_state_dict(state_dict)

        # Done.
        return regressor

    @classmethod
    def create_from_split_state_dict(cls, encoder_state_dict, network_state_dict):
        """
        Instantiate a regressor from a pretrained encoder (scene-agnostic) and a scene-specific head.

        encoder_state_dict: encoder state dictionary
        head_state_dict: scene-specific head state dictionary
        """
        # We simply merge the dictionaries and call the other constructor.
        merged_state_dict = {}

        for k, v in encoder_state_dict.items():
            merged_state_dict[f"encoder.{k}"] = v

        for key in network_state_dict:
            for k, v in network_state_dict[key].items():
                merged_state_dict[f"{key}.{k}"] = v
        # for key in merged_state_dict.keys():
        #     print(key)

        return cls.create_from_state_dict(merged_state_dict)


    def load_encoder(self, encoder_dict_file):
        """
        Load weights into the encoder network.
        """
        self.encoder.load_state_dict(torch.load(encoder_dict_file))

    def get_features(self, inputs):
        return self.encoder(inputs)

    def get_scene_coordinates(self, features):
        return self.heads(features)

    def forward(self, inputs):
        """
        Forward pass.
        """
        features = self.get_features(inputs)
        # B, C, H, W = features.shape
        # X = torch.arange(W, device=torch.device('cuda'))
        # Y = torch.arange(H, device=torch.device('cuda')).reshape(-1, 1)
        # X_enc = torch.zeros((B, 1, H, W), device=torch.device('cuda'))
        # Y_enc = torch.zeros((B, 1, H, W), device=torch.device('cuda'))
        # for i in range(0, H, 2):
        #     X_enc[:,:, i, :] += torch.sin(X * (10000 ** (2 * i / W)))
        #     if i+1>=H:
        #         break
        #     X_enc[:,:, i+1, :] += torch.cos(X * (10000 ** (2 * i / W)))
        # for i in range(0, W, 2):
        #     Y_enc[:,:, :, i] += torch.sin(Y * (10000 ** (2 * i / H))).squeeze(1)
        #     if i+1>=W:
        #         break
        #     Y_enc[:,:, :, i+1] += torch.cos(Y * (10000 ** (2 * i / H))).squeeze(1)
        # X_enc = X_enc.to(dtype=torch.float16)
        # Y_enc = Y_enc.to(dtype=torch.float16)
        pyramid_features = self.se(features)
        features = torch.cat([features,pyramid_features],dim=1)
        # features = self.se(features)
        return self.get_scene_coordinates(features)