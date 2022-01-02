"""csi网络模型"""
import torch.nn as nn

from utils import gs_noise, res_unit, net_standardization


class RMNetStuConfiguration(object):
    """RMStuNet 属性参数"""
    maxtrix_len = 32  # 信号矩阵为方阵，其长度
    channel_num = 2  # 矩阵通道数
    data_length = 2048  # 信号矩阵变成一维向量后的长度 2048 == 2 * 32 *32
    kerner_size = 3
    stride = 1
    padding = 1
    sys_capacity_ratio = 60  # 系统容量倍率（可视化时使用）
    network_name = "RMNetStu"  # 网络名称


class RMNetStu(nn.Module):
    """训练模型三个部分合并封装"""

    def __init__(self, ratio):
        """
        RMStuNet网络模型

        :param ratio: 压缩率
        """
        super(RMNetStu, self).__init__()
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
        self.ratio = ratio  # 压缩率
        self.csi_conv1_combo = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=1, groups=2),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.3)
        )
        self.fc_compress = nn.Linear(2048, 2048 // ratio)

    def forward(self, x):
        """
        压缩: 标准化 ----> 残差（分组卷积） ---> 全连接压缩
        :param x: [batch_size, 2048]
        :return: [batch_size, 2048/ratio]
        """
        x = x.view(-1, 2, 32, 32)  # [batch_size, 2, 32, 32]
        # 分组卷积
        x = res_unit(self.csi_conv1_combo, x)  # [batch_size, 8, 32, 32]
        x = x.view(-1, 2048)  # [batch_size, 2048]
        # 全连接
        output = self.fc_compress(x)  # [batch_size, 2048/ratio]
        return output


class Decoder(nn.Module):
    """解压缩"""

    def __init__(self, ratio):
        super(Decoder, self).__init__()
        self.ratio = ratio
        self.fc_restore = nn.Linear(2048 // ratio, 2048)
        self.group_conv_combo = nn.Sequential(
            Conv2DWrapper(in_channels=2, out_channels=8, kenerl_size=3, stride=1, padding=1, groups=2),
            Conv2DWrapper(in_channels=8, out_channels=16, kenerl_size=3, stride=1, padding=1, groups=4),
            Conv2DWrapper(in_channels=16, out_channels=2, kenerl_size=3, stride=1, padding=1, groups=2)
        )

    def forward(self, x):
        """
        数据解压缩：
        全连接恢复 ---> 残差(分组卷积) ----> -----> 残差(分组卷积)
        :param x: [batch_size, 32*32]
        :return: [batch_size, 32*32]
        """
        # 标准化
        x = net_standardization(x)  # [batch_size, 2048/ratio]
        # 全连接
        x = self.fc_restore(x)  # [batch_size, 2048]
        x = x.view(-1, 2, 32, 32)  # [batch_size, 2, 32, 32]
        # 分组卷积
        x = res_unit(self.group_conv_combo, x)  # [batch_size, 32, 32 ,32]
        output = x.view(-1, 2048)  # [batch_size, 2*32*32]
        return output


class Conv2DWrapper(nn.Module):
    """分组卷积"""

    def __init__(self, in_channels, out_channels, kenerl_size, stride, padding=0, groups=1):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, groups=groups,
                                kernel_size=kenerl_size, stride=stride, padding=padding)
        self.leaky_relu = nn.LeakyReLU(0.3)
        self.bn2d = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.leaky_relu(x)
        x = self.bn2d(x)
        return x


if __name__ == '__main__':
    pass
