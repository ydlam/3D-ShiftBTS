# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ShiftViTBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_div=12,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.BatchNorm3d,
                 outdim = 64,
                 input_resolution=None,
                 input_shape = (64,64,64),):

        super(ShiftViTBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        #mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_hidden_dim = outdim
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        self.n_div = n_div
        #self.a_map = nn.Parameter(torch.ones(dim, input_shape[0], input_shape[1], input_shape[2]))
        #self.b_map = nn.Parameter(torch.ones(dim, input_shape[0], input_shape[1], input_shape[2]))
        #self.c_map = nn.Parameter(torch.ones(dim, input_shape[0], input_shape[1], input_shape[2]))
    def forward(self, x):
        temp = x
        a = self.shift_feat(temp, self.n_div, shiftdis=1)
        b = self.shift_feat(temp, self.n_div, shiftdis=2)
        c = self.shift_feat(temp, self.n_div, shiftdis=3)

        #out = a * self.a_map + b * self.b_map + c * self.c_map
        out = a  + b  + c 
        shortcut = out
        out = shortcut + self.drop_path(self.mlp(self.norm2(out)))

        return out

    def extra_repr(self) -> str:
        return f"dim={self.dim}," \
               f"input_resolution={self.input_resolution}," \
               f"shift percentage={4.0 / self.n_div * 100}%."

    @staticmethod
    def shift_feat(x, n_div, shiftdis = 1):
        B, C, D, H, W = x.shape
        g = C // n_div
        out = torch.zeros_like(x)

        out[:, g * 0:g * 1, :, :, :-shiftdis] = x[:, g * 0:g * 1, :, :, shiftdis:]  # shift left
        out[:, g * 1:g * 2, :, :, shiftdis:] = x[:, g * 1:g * 2, :, :, :-shiftdis]  # shift right

        out[:, g * 0:g * 1, :, :-shiftdis, :] = x[:, g * 0:g * 1, :, shiftdis:, :]  # shift left
        out[:, g * 1:g * 2, :, shiftdis:, :] = x[:, g * 1:g * 2, :, :-shiftdis, :]  # shift right
        out[:, g * 2:g * 3, :-shiftdis, :, :] = x[:, g * 2:g * 3, shiftdis:, :, :]  # shift up
        out[:, g * 3:g * 4, shiftdis:, :, :] = x[:, g * 3:g * 4, :-shiftdis, :, :]  # shift down

        out[:, g * 4:, :, :, :] = x[:, g * 4:, :, :, :]  # no shift
        return out

