# Copyright © Niantic, Inc. 2022.

import logging
import math
import re
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from OverLock.models.overlock import overlock_t, overlock_s, overlock_b

_logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    def __init__(self, model_version='overlock_t', pretrained=True):
        super().__init__()
        # 选择模型类型
        if model_version == 'overlock_t':
            self.model = overlock_t(pretrained=pretrained)
        elif model_version == 'overlock_s':
            self.model = overlock_s(pretrained=pretrained)
        elif model_version == 'overlock_b':
            self.model = overlock_b(pretrained=pretrained)
        else:
            raise ValueError("model_version must be one of ['overlock_t', 'overlock_s', 'overlock_b']")
        
    # 移除分类头
        self.model.head = nn.Identity()
        if hasattr(self.model, 'aux_head'):
            self.model.aux_head = nn.Identity()
        
    def forward(self, x):
        # 获取特征
        features = self.model.forward_features(x)
        return features


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
                 in_channels=768):
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

    OUTPUT_SUBSAMPLE = 4

    def __init__(self, mean, num_head_blocks, use_homogeneous, num_encoder_features=768, model_version='overlock_t'):
        """
        Constructor.

        mean: Learn scene coordinates relative to a mean coordinate (e.g. the center of the scene).
        num_head_blocks: How many extra residual blocks to use in the head (one is always used).
        use_homogeneous: Whether to learn homogeneous or 3D coordinates.
        num_encoder_features: Number of channels output of the encoder network.
        model_version: OverLoCK model version to use ('overlock_t', 'overlock_s', 'overlock_b').
        """
        super(Regressor, self).__init__()

        self.feature_dim = num_encoder_features

        self.encoder = Encoder(model_version=model_version)
        self.heads = Head(mean, num_head_blocks, use_homogeneous, in_channels=self.feature_dim)

    @classmethod
    def create_from_encoder(cls, mean, num_head_blocks, use_homogeneous, model_version='overlock_t'):
        """
        Create a regressor using a pretrained encoder, loading encoder-specific parameters from the state dict.

        encoder_state_dict: pretrained encoder state dictionary.
        mean: Learn scene coordinates relative to a mean coordinate (e.g. the center of the scene).
        num_head_blocks: How many extra residual blocks to use in the head (one is always used).
        use_homogeneous: Whether to learn homogeneous or 3D coordinates.
        model_version: OverLoCK model version to use ('overlock_t', 'overlock_s', 'overlock_b').
        """
        # 根据模型版本确定特征维度
        if model_version == 'overlock_t':
            num_encoder_features = 64  # OverLoCK-overlock_t拼接后的特征维度
        elif model_version == 'overlock_s':
            num_encoder_features = 1536  # OverLoCK-overlock_s拼接后的特征维度
        elif model_version == 'overlock_b':
            num_encoder_features = 3072  # OverLoCK-overlock_b拼接后的特征维度
        else:
            raise ValueError(f"Unsupported OverLoCK model version: {model_version}")

        # Create a regressor.
        _logger.info(f"Creating Regressor using OverLoCK-{model_version} encoder with {num_encoder_features} feature size.")
        regressor = cls(mean, num_head_blocks, use_homogeneous, num_encoder_features, model_version)

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

        # 对于DINOv2,我们使用固定的特征维度
        num_encoder_features = 768//2  # DINOv2拼接后的特征维度

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
    def create_from_split_state_dict(cls, model_version, head_state_dict):
        """
        Instantiate a regressor from a pretrained encoder (scene-agnostic) and a scene-specific head.

        model_version: OverLoCK model version ('overlock_t', 'overlock_s', 'overlock_b')
        head_state_dict: scene-specific head state dictionary
        """
        merged_state_dict = {}
        for k, v in head_state_dict.items():
            merged_state_dict[f"heads.{k}"] = v
        # 根据模型版本确定特征维度
        if model_version == 'overlock_t':
            num_encoder_features = 64  # OverLoCK-overlock_t拼接后的特征维度
        elif model_version == 'overlock_s':
            num_encoder_features = 1536  # OverLoCK-overlock_s拼接后的特征维度
        elif model_version == 'overlock_b':
            num_encoder_features = 3072  # OverLoCK-overlock_b拼接后的特征维度
        else:
            raise ValueError(f"Unsupported OverLoCK model version: {model_version}")

        # Count how many head blocks are in the dictionary.
        pattern = re.compile(r"^heads\.\d+c0\.weight$")
        num_head_blocks = sum(1 for k in merged_state_dict.keys() if pattern.match(k))

        # Whether the network uses homogeneous coordinates.
        use_homogeneous = merged_state_dict["heads.fc3.weight"].shape[0] == 4

        # Create a regressor.
        _logger.info(f"Creating regressor with OverLoCK-{model_version}:"
                     f"\n\tNum head blocks: {num_head_blocks}"
                     f"\n\tHomogeneous coordinates: {use_homogeneous}"
                     f"\n\tEncoder feature size: {num_encoder_features}")
        
        regressor = cls(
            mean=torch.zeros(3),  # 使用零均值
            num_head_blocks=num_head_blocks,
            use_homogeneous=use_homogeneous,
            num_encoder_features=num_encoder_features,
            model_version=model_version
        )

        # Load head weights
        regressor.heads.load_state_dict(head_state_dict)

        return regressor

    def load_encoder(self, encoder_dict_file):
        """
        Load weights into the encoder network.
        """
        # 对于DINOv2,我们不需要加载encoder权重
        # 因为DINOv2模型是通过torch.hub直接加载的
        pass

    def get_features(self, inputs):
        return self.encoder(inputs)

    def get_scene_coordinates(self, features):
        return self.heads(features)

    def forward(self, inputs):
        """
        Forward pass.
        """
        (features_BCHW_4, _, _, _, _) = self.get_features(inputs)
        return self.get_scene_coordinates(features_BCHW_4)