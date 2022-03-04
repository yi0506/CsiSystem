# -*- coding: UTF-8 -*-
"""原始csi_net网络模型的构建、训练、测试并保存结果"""
import torch.nn as nn
from torchsummary import summary

from . import config
from .utils import res_unit, net_standardization, gs_noise


class CSPNetConfiguration(object):
    leak_relu_slope = 0.3  # leaky—relu的负斜率
    sys_capacity_ratio = 45  # 系统容量倍率（可视化时使用）
    maxtrix_len = 32  # 信号矩阵为方阵，其长度
    channel_num = 2  # 矩阵通道数
    data_length = 2048  # 信号矩阵变成一维向量后的长度 2048 == 2 * 32 *32
    kerner_size = 3
    stride = 1
    padding = 1
    network_name = "CSPNet"  # 网络名称


class CSPNet(nn.Module):
    """解压缩"""

    def __init__(self, ratio):
        super(CSPNet, self).__init__()
        self.ratio = ratio  # 信道矩阵通道数
        self.fc_restore = nn.Linear(CSPNetConfiguration.data_length // ratio, CSPNetConfiguration.data_length)
        self.refine_net_1 = PUnit(CSPNetConfiguration.channel_num, CSPNetConfiguration.channel_num)
        self.refine_net_2 = PUnit(CSPNetConfiguration.channel_num, CSPNetConfiguration.channel_num) 
        self.conv2d = nn.Conv2d(in_channels=CSPNetConfiguration.channel_num, out_channels=CSPNetConfiguration.channel_num, 
                                kernel_size=CSPNetConfiguration.kerner_size, stride=CSPNetConfiguration.stride, padding=CSPNetConfiguration.stride)

    def forward(self, y, snr):
        """
        输入：
        
        :param y: [batch_size,  2048/ratio]
        :return: [batch_size,  2048]
        """
        # 是否加入噪声
        if snr is not None:
            y = gs_noise(y, snr)
        # 标准化
        x = net_standardization(y)  # [batch_size, 2048/ratio]
        # 全连接
        x = self.fc_restore(y)  # [batch_size, 2048]
        x = x.view(-1, CSPNetConfiguration.channel_num, CSPNetConfiguration.maxtrix_len, CSPNetConfiguration.maxtrix_len)  # [batch_size, 2, 32, 32]
        # refine_net
        x = res_unit(self.refine_net_1, x)  # [batch_size, 2, 32, 32]
        x = res_unit(self.refine_net_2, x)  # [batch_size, 2, 32, 32]
        x = self.conv2d(x)  # [batch_size, 2, 32, 32]
        output = x.view(-1, CSPNetConfiguration.data_length)
        return output


class PUnit(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.con2d_1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=CSPNetConfiguration.kerner_size,
                                 stride=CSPNetConfiguration.stride, padding=CSPNetConfiguration.padding)
        self.bn2d_1 = nn.BatchNorm2d(8)
        self.leak_relu_1 = nn.LeakyReLU(CSPNetConfiguration.leak_relu_slope)
        self.con2d_2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=CSPNetConfiguration.kerner_size,
                                 stride=CSPNetConfiguration.stride, padding=CSPNetConfiguration.padding)
        self.bn2d_2 = nn.BatchNorm2d(16)
        self.leak_relu_2 = nn.LeakyReLU(CSPNetConfiguration.leak_relu_slope)
        self.con2d_3 = nn.Conv2d(in_channels=16, out_channels=out_channels, kernel_size=CSPNetConfiguration.kerner_size,
                                 stride=CSPNetConfiguration.stride, padding=CSPNetConfiguration.padding)
        self.bn2d_3 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.con2d_1(x)
        x = self.bn2d_1(x)
        x = self.leak_relu_1(x)
        x = self.con2d_2(x)
        x = self.bn2d_2(x)
        x = self.con2d_3(x)
        output = self.bn2d_3(x)
        return output


if __name__ == '__main__':
    model = CSPNet(4).to(config.device)
    summary(model, (2048,))
