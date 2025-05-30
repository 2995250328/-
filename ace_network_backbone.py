# Copyright © Niantic, Inc. 2022.

import logging
import math
import re
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

_logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    """
    DINOv2 encoder, used to extract features from the input images.
    """
    def __init__(self, out_channels=768, model_version='vits14'):
        super(Encoder, self).__init__()
        
        self.out_channels = out_channels
        
        # 获取dinov2目录的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dinov2_dir = os.path.join(current_dir, 'dinov2')
        
        # 加载DINOv2模型
        self.dinov2 = torch.hub.load(dinov2_dir, f'dinov2_{model_version}', source='local')
        
        # 冻结DINOv2参数
        for param in self.dinov2.parameters():
            param.requires_grad = False
            
        # 特征维度
        if model_version == 'vits14':
            self.feat_dim = 384  # DINOv2-vits14的特征维度
        elif model_version == 'vitb14':
            self.feat_dim = 768  # DINOv2-vitb14的特征维度
        elif model_version == 'vitl14':
            self.feat_dim = 1024  # DINOv2-vitl14的特征维度
        elif model_version == 'vitg14':
            self.feat_dim = 1536  # DINOv2-vitg14的特征维度
        else:
            raise ValueError(f"Unsupported DINOv2 model version: {model_version}")
        
        # DINOv2的patch size
        self.patch_size = 14

    def get_attention_weights(self, x):
        """
        获取DINOv2的注意力权重
        Args:
            x: 输入图像 [B, C, H, W]
        Returns:
            attention_weights: 注意力权重 [B, num_heads, num_patches+1, num_patches+1]
        """
        # 获取注意力权重
        last_block = self.dinov2.blocks[-1]
        last_attention_module = last_block.attn
        last_block_attention_map = last_attention_module.attn_weights
        aggregated_attention_map = last_block_attention_map.mean(dim=1)
        return aggregated_attention_map[0,:,:]

    def _adjust_image_size(self, x):
        """
        调整输入图像尺寸，使其是patch_size的整数倍
        """
        B, C, H, W = x.shape
        
        # 计算需要padding的像素数
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        
        if pad_h > 0 or pad_w > 0:
            # 使用反射填充
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            
        return x

    def forward(self, x):
        # 调整图像尺寸
        # x = self._adjust_image_size(x)
        
        # 获取DINOv2特征
        features_dict = self.dinov2.forward_features(x)
        
        # 获取patch特征和cls特征
        patch_features = features_dict['x_norm_patchtokens']  # [B, N, C]
        cls_features = features_dict['x_norm_clstoken']  # [B, 1, C]
        
        # 获取注意力权重
        attention_weights = self.get_attention_weights(x)  # [B, num_heads, num_patches+1, num_patches+1]
        
        # 将cls特征广播到与patch特征相同的空间维度
        B, N, C = patch_features.shape
        H = W = int(math.sqrt(N))
        cls_features = cls_features.expand(B, N, -1)  # [B, N, C]
        
        # 在特征维度上拼接
        concat_features = torch.cat([patch_features, cls_features], dim=-1)  # [B, N, 2C]
        
        # 重塑为空间特征图
        concat_features = concat_features.transpose(1, 2).reshape(B, 2*C, H, W)
        
        return concat_features, attention_weights


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

    OUTPUT_SUBSAMPLE = 8

    def __init__(self, mean, num_head_blocks, use_homogeneous, num_encoder_features=768, model_version='vits14'):
        """
        Constructor.

        mean: Learn scene coordinates relative to a mean coordinate (e.g. the center of the scene).
        num_head_blocks: How many extra residual blocks to use in the head (one is always used).
        use_homogeneous: Whether to learn homogeneous or 3D coordinates.
        num_encoder_features: Number of channels output of the encoder network.
        model_version: DINOv2 model version to use ('vits14', 'vitb14', 'vitl14', 'vitg14').
        """
        super(Regressor, self).__init__()

        self.feature_dim = num_encoder_features

        self.encoder = Encoder(out_channels=self.feature_dim, model_version=model_version)
        self.heads = Head(mean, num_head_blocks, use_homogeneous, in_channels=self.feature_dim)

    @classmethod
    def create_from_encoder(cls, mean, num_head_blocks, use_homogeneous, model_version='vits14'):
        """
        Create a regressor using a pretrained encoder, loading encoder-specific parameters from the state dict.

        encoder_state_dict: pretrained encoder state dictionary.
        mean: Learn scene coordinates relative to a mean coordinate (e.g. the center of the scene).
        num_head_blocks: How many extra residual blocks to use in the head (one is always used).
        use_homogeneous: Whether to learn homogeneous or 3D coordinates.
        model_version: DINOv2 model version to use ('vits14', 'vitb14', 'vitl14', 'vitg14').
        """
        # 根据模型版本确定特征维度
        if model_version == 'vits14':
            num_encoder_features = 768  # DINOv2-vits14拼接后的特征维度
        elif model_version == 'vitb14':
            num_encoder_features = 1536  # DINOv2-vitb14拼接后的特征维度
        elif model_version == 'vitl14':
            num_encoder_features = 2048  # DINOv2-vitl14拼接后的特征维度
        elif model_version == 'vitg14':
            num_encoder_features = 3072  # DINOv2-vitg14拼接后的特征维度
        else:
            raise ValueError(f"Unsupported DINOv2 model version: {model_version}")

        # Create a regressor.
        _logger.info(f"Creating Regressor using DINOv2-{model_version} encoder with {num_encoder_features} feature size.")
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
        num_encoder_features = 768  # DINOv2拼接后的特征维度

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
    def create_from_split_state_dict(cls, encoder_state_dict, head_state_dict):
        """
        Instantiate a regressor from a pretrained encoder (scene-agnostic) and a scene-specific head.

        encoder_state_dict: encoder state dictionary
        head_state_dict: scene-specific head state dictionary
        """
        # 对于DINOv2,我们不需要合并encoder的state_dict
        # 因为DINOv2模型是通过torch.hub直接加载的
        
        # 我们只需要使用head的state_dict
        return cls.create_from_state_dict(head_state_dict)

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
        features = self.get_features(inputs)
        return self.get_scene_coordinates(features)