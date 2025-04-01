# Copyright © Niantic, Inc. 2022.

import logging
import math
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from superpoint import SuperPointNet

_logger = logging.getLogger(__name__)

#@save 加性和点积
class Attention(nn.Module):
    """加性和点积注意力"""
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.softmax = nn.Softmax(dim=1)

    # queries的形状：(查询的个数，d)
    # keys的形状：(“键－值”对的个数，d)
    # values的形状：(“键－值”对的个数，值的维度)
    def forward(self, queries, keys, values ):
        d = queries.shape[-1]
        scores = torch.matmul(queries,keys.transpose(0,1))/math.sqrt(d)
        scores = self.softmax(scores)
        return torch.matmul(scores,values)
#@save
class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size=512, query_size=512, value_size=512, num_hiddens=512,
                 num_heads=8, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = Attention()
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias,dtype=torch.float16)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias,dtype=torch.float16)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias,dtype=torch.float16)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias,dtype=torch.float16)
     
    def transpose_qkv(self, X, num_heads):
        """
        为了多注意力头的并行计算而变换形状，适配二维输入数据(N, C)。
        输入:
            X: 形状 (N, C)
            num_heads: 注意力头的数量
        输出:
            转换后的形状为 (num_heads * N, C / num_heads)
        """
        # 检查输入维度是否能被分为多头
        num_features = X.shape[-1]  # C
        assert num_features % num_heads == 0, "特征维度 C 必须能被 num_heads 整除"
        # 计算每个头的特征维度
        head_dim = num_features // num_heads
        # 改变形状: (N, C) -> (N, num_heads, head_dim)
        X = X.reshape(X.shape[0], num_heads, head_dim)

        # 调整形状: (N, num_heads, head_dim) -> (num_heads * N, head_dim)
        return X.permute(1, 0, 2).reshape(-1, head_dim)


    #@save
    def transpose_output(self, X, num_heads):
        """
        逆转 transpose_qkv 的变换，适配二维输入数据。
        输入:
            X: 形状 (num_heads * N, head_dim)
            num_heads: 注意力头的数量
        输出:
            形状为 (N, C)
        """
        # 计算原始维度
        head_dim = X.shape[-1]
        N = X.shape[0] // num_heads

        # 恢复形状: (num_heads * N, head_dim) -> (num_heads, N, head_dim)
        X = X.reshape(num_heads, N, head_dim)

        # 调整形状: (num_heads, N, head_dim) -> (N, num_heads, head_dim)
        X = X.permute(1, 0, 2)

        # 合并最后两维: (N, num_heads, head_dim) -> (N, C)
        return X.reshape(N, -1) 

    def forward(self, queries, keys, values):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)
        queries = self.transpose_qkv(self.W_q(queries), self.num_heads)
        keys = self.transpose_qkv(self.W_k(keys), self.num_heads)
        values = self.transpose_qkv(self.W_v(values), self.num_heads)

        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = self.transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
#@save
class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input=512, ffn_num_hiddens=512, ffn_num_outputs=512,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens,dtype=torch.float16)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs,dtype=torch.float16)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

#@save
class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.ln = nn.LayerNorm(normalized_shape,dtype=torch.float16)

    def forward(self, X, Y):
        return self.ln(Y + X)

#@save
class EncoderBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, key_size=512, query_size=512, value_size=512, num_hiddens=512,
                 norm_shape=512, ffn_num_input=512, ffn_num_hiddens=512, num_heads=8,
                 use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, 
            use_bias)
        self.addnorm1 = AddNorm(norm_shape)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape)

    def forward(self, X1,X2):
        Y = self.addnorm1(X1, self.attention(X1, X2, X2))
        return self.addnorm2(Y, self.ffn(Y))

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens,  max_len=1000):
        super(PositionalEncoding, self).__init__()
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return X
#@save
class TransformerEncoder(EncoderBlock):
    """Transformer编码器"""
    def __init__(self, key_size=512, query_size=512, value_size=512,
                 num_hiddens=512, norm_shape=512, ffn_num_input=512, ffn_num_hiddens=512,
                 num_heads=8, num_layers=4, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        #self.pos_encoding = PositionalEncoding(num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, use_bias))

    def forward(self, X1,X2, *args):
        # 因为位置编码值在-1和1之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，
        # 然后再与位置编码相加。
        #X1 = self.pos_encoding(X1 * math.sqrt(self.num_hiddens))
        for i, blk in enumerate(self.blks):
            X = blk(X1,X2)
        return X

class PtsAndDesc(nn.Module):
    def __init__(self):
        super(PtsAndDesc,self).__init__()        
        self.linear1 = nn.Linear(2,256,bias=True,dtype=torch.float16)
        self.linear2 = nn.Linear(256,512,bias=True,dtype=torch.float16)
        self.BN = nn.BatchNorm1d(num_features=512,dtype=torch.float16)

    def forward(self,pts,desc):
        x = self.linear1(pts)+desc
        x = self.linear2(x)
        x = self.BN(x)
        return x

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
        self.ptsanddesc = PtsAndDesc()
        self.trans_encoder = TransformerEncoder(num_layers=4)
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

    def forward(self, inputs ,superpoint):
        """
        Forward pass.
        """
        features = self.get_features(inputs)
        #print(features.shape)
        image_BHW = inputs.squeeze(1)
        image_HW = image_BHW.squeeze(0)
        image_HW_np = image_HW.cpu().numpy().astype(np.float32)
        pts, desc, _ = superpoint.run(image_HW_np)
        pts_tensor = torch.from_numpy(pts)
        desc_tensor = torch.from_numpy(desc)
        pts_tensor = pts_tensor.permute(1,0).to(torch.device("cuda"),non_blocking=True,dtype=features.dtype)
        desc_tensor = desc_tensor.permute(1,0).to(torch.device("cuda"),non_blocking=True,dtype=features.dtype)
        #print(pts_tensor.shape)
        ptsanddesc = self.ptsanddesc(pts_tensor[:,:2],desc_tensor)
        #print(ptsanddesc.shape)
        def normalize_shape(tensor_in):
            """Bring tensor from shape BxCxHxW to NxC"""
            return tensor_in.transpose(0, 1).flatten(1).transpose(0, 1)
        B, C, H, W = features.shape
        features_NC = normalize_shape(features)
        #print(features_NC.shape)
        features_NC_crossAttention = self.trans_encoder(features_NC,ptsanddesc)
        #print(features_NC_crossAttention.shape)
        features_bCHW = features_NC_crossAttention[None, None, ...].view(-1, H, W, C).permute(0, 3, 1, 2)
        #print(features_bCHW.shape)

        return self.get_scene_coordinates(features_bCHW)
