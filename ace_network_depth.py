# Copyright © Niantic, Inc. 2022.

import logging
import math
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from superpoint import SuperPointNet
from depth_anything_v2.dpt import DepthAnythingV2

_logger = logging.getLogger(__name__)

class RelativeDepthLoss(nn.Module):
    def __init__(self,weight):
        super().__init__()
        self.weight = weight
        # 初始化Sobel梯度算子
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.sobel_x.weight.data = sobel_kernel_x.view(1, 1, 3, 3)
        self.sobel_y.weight.data = sobel_kernel_y.view(1, 1, 3, 3)
        
    def compute_pairwise_diff(self, depth1, depth2, mask):
        """
        计算成对像素深度差矩阵
        参数:
            depth1: [H,W] 深度图1
            depth2: [H,W] 深度图2 
            mask:   [H,W] 有效像素掩膜
        返回:
            diff_matrix: [H*W,H*W] 成对差异矩阵
        """
        H, W = depth1.shape
        flat_depth1 = depth1.view(-1)  # [HW]
        flat_depth2 = depth2.view(-1)
        flat_mask = mask.view(-1)
        
        # 计算元素间差值
        diff1 = flat_depth1.unsqueeze(1) - flat_depth1.unsqueeze(0)  # [HW,HW]
        diff2 = flat_depth2.unsqueeze(1) - flat_depth2.unsqueeze(0)  # [HW,HW]
        diff = diff1-diff2
        # 两幅图之间直接进行相对深度计算
        # diff = flat_depth1.unsqueeze(1) - flat_depth2.unsqueeze(0)  # [HW,HW]
        valid_mask = flat_mask.unsqueeze(1) & flat_mask.unsqueeze(0)  # 有效对掩膜
        
        # 空间距离归一化
        coord = torch.stack(torch.meshgrid(
            torch.arange(H), torch.arange(W), indexing='ij'
        ), -1).float().to(depth1.device)
        coord = coord.view(-1, 2)  # [HW,2]
        spatial_dist = torch.norm(
            coord.unsqueeze(1) - coord.unsqueeze(0), 
            dim=-1
        ) + 1e-3  # [HW,HW]
        spatial_dist = spatial_dist.view(H*W,H*W)
        
        return diff / spatial_dist, valid_mask

    def forward(self, pred_depth, gt_depth,valid_mask):
        """
        前向计算总损失
        参数:
            pred_depth: 预测深度图 [B,1,H,W]
            gt_depth:   真值深度图 [B,1,H,W]
        返回:
            total_loss: 总损失值
        """       
        # 基础深度差异损失(网页6像素差异损失)
        # depth_diff = (pred_depth - gt_depth).abs()
        # depth_loss = (depth_diff * valid_mask).sum() / valid_mask.sum()
        
        # 多尺度梯度一致性损失(网页1/网页4梯度约束)
        grad_x_pred = self.sobel_x(pred_depth.unsqueeze(0))
        grad_y_pred = self.sobel_y(pred_depth.unsqueeze(0))
        grad_x_gt = self.sobel_x(gt_depth.unsqueeze(0))
        grad_y_gt = self.sobel_y(gt_depth.unsqueeze(0))
        grad_diff = (grad_x_pred - grad_x_gt).abs() + (grad_y_pred - grad_y_gt).abs()
        grad_loss = (grad_diff * valid_mask).sum() / valid_mask.sum()
        
        # 成对相对差异损失(网页6相对排序损失扩展)
        pair_diff, pair_mask = self.compute_pairwise_diff(pred_depth, gt_depth, valid_mask)
        pair_loss = (pair_diff.abs() * pair_mask).sum() / math.sqrt(pair_mask.sum()+1e-5)
        
        # 总损失组合(网页5损失权重思想)
        total_loss = (1-self.weight)*grad_loss + self.weight*pair_loss
        return math.log(total_loss+1)

def compute_patch_indices(coords, H, W, patch_size=8):
        """
        输入:
        coords: [N, 2] 的张量，每一行为 [x, y]（整数坐标），表示 patch 的中心坐标
        H, W: 图像高度和宽度
        patch_size: 每个 patch 的尺寸（默认 8）
        输出:
        final_indices: [N] 的张量，每个元素为该原始坐标对应的 patch 序号
        """
        # 1. 去除重复的坐标
        unique_coords, inverse_indices = torch.unique(coords, return_inverse=True, dim=0)
        # print(unique_coords)
        
        # 2. 根据每个唯一坐标计算其所在的网格位置
        # 行号（row）由 y 值决定，列号（col）由 x 值决定
        rows = unique_coords[:, 1] // patch_size   # y // patch_size
        cols = unique_coords[:, 0] // patch_size   # x // patch_size
        
        # 图像水平上有多少个 patch
        num_cols = W // patch_size
        
        # 计算网格中该 patch 的序号（从 0 开始），序号 = row * num_cols + col
        patch_idx = rows * num_cols + cols
        # print(patch_idx)
        
        # 返回唯一的 patch 序号
        return patch_idx

def find_neighbors_with_confidence(coords, H, W, patch_size, include_diagonal=8):
    """
    Find neighbors for a 3D coordinate tensor with confidence values, filter duplicates
    based on confidence, and return the results.

    Args:
        coords (np.ndarray): Shape (N, 3), where each row contains (x, y, confidence).
        H (int): Image height.
        W (int): Image width.
        patch_size (int): Size of the patch.
        include_diagonal (bool): Whether to include diagonal neighbors.

    Returns:
        np.ndarray: Filtered neighbors with shape (M, 4), where each row contains
                    (x, y, confidence, original_index).
    """
    # Define neighbor offsets
    offsets =[(0,0)]
    if include_diagonal==4:
        offsets += [
            (-1, 0), (1, 0), (0, -1), (0, 1)  # Up, Down, Left, Right
        ]
    elif include_diagonal==8:
        offsets += [
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # Diagonal neighbors
        ]

    # Initialize results storage
    neighbors = []
    #print(f'{H}   {W}')

    for idx, (x, y, confidence) in enumerate(coords):
        for dx, dy in offsets:
            nx, ny = x + dx * patch_size, y + dy * patch_size
            #print(f'{nx},{ny}')

            # Check if the neighbor is within bounds
            if 0 <= nx < W and 0 <= ny < H:
                neighbors.append((nx, ny, confidence, idx))
    # Convert to numpy array
    neighbors = np.array(neighbors, dtype=np.float32)
    #print(neighbors.shape)

    # Sort neighbors by coordinates and confidence (descending)
    neighbors = neighbors[np.lexsort((-neighbors[:, 2], neighbors[:, 1], neighbors[:, 0]))]

    # Remove duplicates by keeping the highest confidence
    unique_neighbors = []
    seen_coords = set()

    for x, y, conf, orig_idx in neighbors:
        coord_key = (x, y)
        if coord_key not in seen_coords:
            unique_neighbors.append((x, y, conf, orig_idx))
            seen_coords.add(coord_key)

    return np.array(unique_neighbors, dtype=np.float32)

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

    def __init__(self, mean, num_head_blocks, use_homogeneous,
                 num_encoder_features=512):
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
        self.superpoint = SuperPointNet()
        self.heads = Head(mean, num_head_blocks, use_homogeneous, in_channels=self.feature_dim)

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
        regressor = cls(mean, num_head_blocks, use_homogeneous,num_encoder_features)

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
        regressor = cls(mean, num_head_blocks, use_homogeneous,num_encoder_features)

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
                print(k)
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
        # def normalize_shape(tensor_in):
        #     """Bring tensor from shape BxCxHxW to BxNxC"""
        #     return tensor_in.flatten(2).transpose(1, 2)
        # B,_,H,W = inputs.shape
        # features = self.get_features(inputs)
        # _,_,fH,fW = features.shape
        # #print(features.shape)
        # image_BHW = inputs.squeeze(1)
        # image_HW = image_BHW.squeeze(0)
        # image_HW_np = image_HW.cpu().numpy().astype(np.float32)
        # pts, desc, _ = superpoint.run(image_HW_np)
        # pts_tensor = torch.from_numpy(pts)
        # # desc_tensor = torch.from_numpy(desc)
        # pts_tensor = pts_tensor.permute(1,0).to(torch.device("cuda"),non_blocking=True,dtype=features.dtype)
        # pts_tensor[:,:2] = (pts_tensor[:,:2]//self.OUTPUT_SUBSAMPLE)*self.OUTPUT_SUBSAMPLE + \
        #                     torch.tensor(4,device=pts_tensor.device,dtype=pts_tensor.dtype)
        # pts_np = pts_tensor.cpu().numpy().astype(np.float32)
        # position_confidence_idx = find_neighbors_with_confidence(pts_np,H,W,8,4)
        # position_confidence_idx = torch.from_numpy(position_confidence_idx)
        # patch_idx = compute_patch_indices(position_confidence_idx[:,:2],H,W)
        # patch_idx = torch.tensor(patch_idx,device=inputs.device,dtype=torch.int)
        # scene_coordinates_BCHW = self.get_scene_coordinates(features)
        # scene_coordinates_BNC = normalize_shape(scene_coordinates_BCHW)
        # scene_coordinates_BNC = scene_coordinates_BNC[:,patch_idx,:]
        
        # return scene_coordinates_BCHW,scene_coordinates_BNC,patch_idx,fH,fW
        features = self.get_features(inputs)
        return self.get_scene_coordinates(features)
