import torch
import torch.nn as nn
from torch.nn import functional as F
from models.SPACH.shiftvit import *
from models.BiTrUnet.layers import ConvBnRelu, UBlock, conv1x1, UBlockCbam, CBAM
from models.BiTrUnet.unet import Att_EquiUnet
from models.BiTrUnet.Myshiftvit import *

class DeUp_Cat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeUp_Cat, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)  # decrease half size
        self.conv3 = nn.Conv3d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x, prev):
        x1 = self.conv1(x)
        y = self.conv2(x1)
        # y = y + prev
        y = torch.cat((prev, y), dim=1)
        y = self.conv3(y)
        return y


class Cat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Cat, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv3d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x, prev):
        x1 = self.conv1(x)
        y = self.conv2(x1)
        # y = y + prev
        y = torch.cat((prev, y), dim=1)
        y = self.conv3(y)
        return y

class DeBlock(nn.Module):
    def __init__(self, in_channels):
        super(DeBlock, self).__init__()

        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = x1 + x

        return x1

class My_Net(nn.Module):
    def __init__(self, in_dim, embedding_dim):
        super(My_Net, self).__init__()

        self.in_dim = in_dim
        self.embedding_dim = embedding_dim

        self.Softmax = nn.Softmax(dim=1)


        self.DeUp5 = DeUp_Cat(in_channels=self.embedding_dim * 8, out_channels=self.embedding_dim * 4)  # 128
        self.DeBlock5 = DeBlock(in_channels=self.embedding_dim * 4)  # 128
        #self.Block5_2 = MyShiftViTBlock(dim=self.embedding_dim * 4, input_shape=(16,16,16))  # 128


        self.Cat4 = DeUp_Cat(in_channels=self.embedding_dim * 4, out_channels=self.embedding_dim * 2)
        self.Block4 = DeBlock(in_channels=self.embedding_dim * 2)
        #self.Block4_2 = MyShiftViTBlock(dim=self.embedding_dim * 2, input_shape=(32,32,32))  # 128

        self.DeUp4 = DeUp_Cat(in_channels=self.embedding_dim  * 2, out_channels=self.embedding_dim)  # 64
        self.DeBlock4 = DeBlock(in_channels=self.embedding_dim)
        #self.DeBlock4_2 = MyShiftViTBlock(dim=self.embedding_dim, input_shape=(64,64,64))  # 128

        self.DeUp3 = DeUp_Cat(in_channels=self.embedding_dim, out_channels=16)  # 32
        self.DeBlock3 = DeBlock(in_channels=16)
        self.DeBlock3_2 = MyShiftViTBlock(dim=16, input_shape=(128,128,128))  # 128

        self.endconv = nn.Conv3d(16, 4, kernel_size=1)  # 16

        self.encoder = ShiftViT(embed_dim=self.embedding_dim, patch_size=2, in_chans=16, depths=(2, 2, 18, 2), mlp_ratio=4, drop_path_rate=0.5, n_div=16)
        self.Unet = Att_EquiUnet(inplanes = 16, num_classes = 4 , width = 16)


    def forward(self, x):
        x1_1 = self.Unet(x)


        y_list = self.encoder(x1_1)


        out3 = self.DeUp5(y_list[3], y_list[2])  # (1, 128, 32, 32, 32)
        out3 = self.DeBlock5(out3)
        #out3 = self.Block5_2(out3)

        out2 = self.Cat4(out3, y_list[1])  # (1, 64, 32, 32, 32)
        out2 = self.Block4(out2)
        #out2 = self.Block4_2(out2)

        out1 = self.DeUp4(out2, y_list[0])  # (1, 64, 32, 32, 32)
        out1 = self.DeBlock4(out1)
        #out1 = self.DeBlock4_2(out1)

        out = self.DeUp3(out1, x1_1)  # (1, 32, 64, 64, 64)
        out = self.DeBlock3(out)
        out = self.DeBlock3_2(out)
        
        y = self.endconv(out)  # (1, 4, 128, 128, 128)
        y = self.Softmax(y)
        return y


if __name__ == "__main__":
    # from thop import profile
    device = torch.device('cuda')
    image_size = 128
    x = torch.rand((1, 4, 128, 128, 128))
    print("x size: {}".format(x.size()))
    model = My_Net(in_dim=4, embedding_dim=64)
    # flops, params = profile(model, inputs=(x,))
    # print("***********")
    # print(flops, params)
    # print("***********")
    out = model(x)
    print("out size: {}".format(out.size()))
