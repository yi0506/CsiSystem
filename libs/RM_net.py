"""csi网络模型"""
import torch.nn as nn
from torchsummary import summary

import config
from utils import gs_noise, res_unit, net_normalize


class RMNetConfiguration(object):
    """RMNet 属性参数"""
    maxtrix_len = 32  # 信号矩阵为方阵，其长度
    channel_num = 2  # 矩阵通道数
    data_length = 2048  # 信号矩阵变成一维向量后的长度 2048 == 2 * 32 *32
    kerner_size = 3
    stride = 1
    padding = 1
    sys_capacity_ratio = 60  # 系统容量倍率（可视化时使用）
    network_name = "RMNet"  # 网络名称


class RMNet(nn.Module):
    """训练模型三个部分合并封装"""

    def __init__(self, ratio):
        """
        RMNet网络模型
        
        :param ratio: 压缩率
        """
        super(RMNet, self).__init__()
        self.encoder = Encoder(ratio)
        self.decoder = Decoder(ratio)

    def forward(self, input_, snr):
        encoder_output = self.encoder(input_)
        add_noise = gs_noise(encoder_output, snr)
        decoder_output = self.decoder(add_noise)
        return decoder_output


class Encoder(nn.Module):
    """MS压缩"""

    def __init__(self, ratio):
        super(Encoder, self).__init__()
        self.ratio = ratio
        self.conv2d_1 = nn.Conv2d(RMNetConfiguration.channel_num, 8, kernel_size=RMNetConfiguration.kerner_size,
                                  stride=RMNetConfiguration.stride, padding=RMNetConfiguration.padding)
        self.group_conv_combo_1 = nn.Sequential(
            Conv2DWrapper(8, 16, RMNetConfiguration.kerner_size, RMNetConfiguration.stride, RMNetConfiguration.padding, 4),
            Conv2DWrapper(16, 8, RMNetConfiguration.kerner_size, RMNetConfiguration.stride, RMNetConfiguration.padding, 4),
        )
        self.group_conv_combo_2 = nn.Sequential(
            Conv2DWrapper(8, 16, RMNetConfiguration.kerner_size, RMNetConfiguration.stride, RMNetConfiguration.padding, 4),
            Conv2DWrapper(16, 8, RMNetConfiguration.kerner_size, RMNetConfiguration.stride, RMNetConfiguration.padding, 2)
        )
        self.conv2d_2 = nn.Conv2d(8, RMNetConfiguration.channel_num, kernel_size=RMNetConfiguration.kerner_size,
                                stride=RMNetConfiguration.stride, padding=RMNetConfiguration.padding)
        self.fc_compress = LinearWrapper(RMNetConfiguration.data_length, RMNetConfiguration.data_length // self.ratio)

    def forward(self, input_):
        """
        压缩：标准化 ----> 全连接 ----> 残差（分组卷积） ---> 残差（分组卷积） ---> 全连接压缩
        :param input_: [batch_size, 2048]
        :return: [batch_size, 2048/ratio]
        """
        # 标准化
        x = net_normalize(input_)  # [batch_size, 2048]
        x = x.view(-1, RMNetConfiguration.channel_num, RMNetConfiguration.maxtrix_len, RMNetConfiguration.maxtrix_len)  # [batch_size, 2, 32, 32]
        # 卷积
        x = self.conv2d_1(x)  # [batch_size, 8, 32, 32]
        # 分组卷积
        x = res_unit(self.group_conv_combo_1, x)  # [batch_size, 8, 32, 32]
        x = res_unit(self.group_conv_combo_2, x)  # [batch_size, 8, 32, 32]
        x = self.conv2d_2(x)  # [batch_size, 2, 32, 32]
        x = x.view(-1, RMNetConfiguration.data_length)  # [batch_size, 2048]
        # 全连接
        output = self.fc_compress(x)  # [batch_size, 2048/ratio]
        return output


class Decoder(nn.Module):
    """解压缩"""

    def __init__(self, ratio):
        super(Decoder, self).__init__()
        self.ratio = ratio
        self.restore_fc = LinearWrapper(RMNetConfiguration.data_length // ratio,  RMNetConfiguration.data_length)
        self.conv2d_1 = nn.Conv2d(RMNetConfiguration.channel_num, 8, RMNetConfiguration.kerner_size,
                                  RMNetConfiguration.stride, RMNetConfiguration.padding)
        self.deep_conv_combo = nn.Sequential(
            Conv2DWrapper(8, 8, RMNetConfiguration.kerner_size, RMNetConfiguration.stride, RMNetConfiguration.padding, 4),
            Conv2DWrapper(8, 8, RMNetConfiguration.kerner_size, RMNetConfiguration.stride, RMNetConfiguration.padding, 4)
        )
        self.group_conv_combo = nn.Sequential(
            Conv2DWrapper(8, 32, RMNetConfiguration.kerner_size, RMNetConfiguration.stride, RMNetConfiguration.padding, 8),
            Conv2DWrapper(32, 8, RMNetConfiguration.kerner_size, RMNetConfiguration.stride, RMNetConfiguration.padding, 8)
        )
        self.deep_separate_combo_1 = nn.Sequential(
            DeepSeparateConv2DWrapper(8, 64, RMNetConfiguration.kerner_size, RMNetConfiguration.stride, RMNetConfiguration.padding),
            Conv2DWrapper(64, 8, RMNetConfiguration.kerner_size, RMNetConfiguration.stride, RMNetConfiguration.padding)
        )
        self.deep_separate_combo_2 = nn.Sequential(
            DeepSeparateConv2DWrapper(8, 64, RMNetConfiguration.kerner_size, RMNetConfiguration.stride, RMNetConfiguration.padding),
            Conv2DWrapper(64, 8, RMNetConfiguration.kerner_size, RMNetConfiguration.stride, RMNetConfiguration.padding)
        )
        self.conv2d_2 = nn.Conv2d(8, 2, RMNetConfiguration.kerner_size, RMNetConfiguration.stride, RMNetConfiguration.padding)

    def forward(self, x):
        """
        数据解压缩：
        残差（分组卷积） ----> 残差（深度可分离卷积）-----> 残差（深度可分离卷积） -----> 全连接恢复
        :param x: [batch_size, 32*32]
        :return: [batch_size, 32*32]
        """
        # 标准化
        x = net_normalize(x)  # [batch_size, 2048/ratio]
        # 全连接
        x = self.restore_fc(x)  # [batch_size, 2048]
        x = x.view(-1, RMNetConfiguration.channel_num, RMNetConfiguration.maxtrix_len, RMNetConfiguration.maxtrix_len)  # [batch_size, 2, 32, 32]
        # 卷积
        x = self.conv2d_1(x)  # [batch_size, 8, 32, 32]
        # 深度卷积
        x = res_unit(self.deep_conv_combo, x)  # [batch_size, 8, 32 ,32]
        # 分组卷积
        x = res_unit(self.group_conv_combo, x)  # [batch_size, 32, 32 ,32]
        # 深度可分卷积
        x = res_unit(self.deep_separate_combo_1, x)  # [batch_size, 32, 32 ,32]
        x = res_unit(self.deep_separate_combo_2, x)  # [batch_size, 2, 32 ,32]
        x = self.conv2d_2(x)
        output = x.view(-1, RMNetConfiguration.data_length)  # [batch_size, 2*32*32]
        return output
    

class DeepSeparateConv2DWrapper(nn.Module):
    """深度可分离卷积"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, groups=in_channels, 
                                  kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2d_2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.elu = nn.ELU(True)
        self.bn2d = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.elu(x)
        x = self.bn2d(x)
        return x


class LinearWrapper(nn.Module):
    """全连接层压缩"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.elu = nn.ELU(True)
        self.bn2d = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = self.linear(x)
        x = self.elu(x)
        x = self.bn2d(x)
        return x


class Conv2DWrapper(nn.Module):
    """分组卷积"""

    def __init__(self, in_channels, out_channels, kenerl_size, stride, padding=0, groups=1):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, groups=groups, 
                                kernel_size=kenerl_size, stride=stride, padding=padding)
        self.elu = nn.ELU(True)
        self.bn2d = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.elu(x)
        x = self.bn2d(x)
        return x


if __name__ == '__main__':
    pass
