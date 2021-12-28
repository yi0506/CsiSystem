"""csi网络模型"""
import torch.nn as nn

from utils import gs_noise, res_unit, net_standardization


class RMNetConfiguration(object):
    """RMNet 属性参数"""
    maxtrix_len = 32  # 信号矩阵为方阵，其长度
    channel_num = 2  # 矩阵通道数
    data_length = 2048  # 信号矩阵变成一维向量后的长度 2048 == 2 * 32 *32
    kerner_size = 3
    stride = 1
    padding = 1
    sys_capacity_ratio = 60  # 系统容量倍率（可视化时使用）
    leak_relu_slope = 0.3
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
    """压缩"""

    def __init__(self, ratio):
        super(Encoder, self).__init__()
        self.ratio = ratio  # 压缩率
        self.csi_conv1_combo = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.3)
        )
        self.fc_compress = nn.Linear(2048, 2048 // ratio)

    def forward(self, input_):
        """
        编码器: 卷积 ---> 全连接压缩

        """
        # 卷积
        x = input_.view(-1, 2, 32, 32)
        x = self.csi_conv1_combo(x)  # [batch_size, 2, 32, 32]
        # 全连接
        x = x.view(-1, 2048)  # [batch_size, 2048]
        output = self.fc_compress(x)  # [batch_size, 2048/ratio]
        return output


class Decoder(nn.Module):
    """解压缩"""

    def __init__(self, ratio):
        super(Decoder, self).__init__()
        self.ratio = ratio  # 信道矩阵通道数
        self.fc_restore = nn.Linear(2048 // ratio, 2048)
        self.refine_net_1 = RefineNet(2, 2)
        self.refine_net_2 = RefineNet(2, 2)
        self.conv2d_combo = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2),
            nn.Sigmoid()
        )

    def forward(self, encoder_output):
        """
        由于训练数据形状为[batch_size, 2048]，
        因此最后需要reshape操作：[batch_size, 2, 32, 32] -----> [batch_size,  2048]，变为原来的形状

        数据解压缩：
        全连接 ----> reshape ----> 残差（refine_net） ----> 残差（refine_net） -----> 卷积  -----> reshape

        :param encoder_output: [batch_size,  2048/ratio]
        :return: [batch_size, 2048]
        """

        # 全连接
        x = net_standardization(encoder_output)
        x = self.fc_restore(x)  # [batch_size, 2048]
        x = x.view(-1, 2, 32, 32)  # [batch_size, 2, 32, 32]
        # refine_net
        x = res_unit(self.refine_net_1, x)  # [batch_size, 2, 32, 32]
        x = res_unit(self.refine_net_2, x)  # [batch_size, 2, 32, 32]
        # 卷积
        x = self.conv2d_combo(x)  # [batch_size, 2, 32, 32]
        output = x.view(-1, 2048)
        return output


class RefineNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.con2d_1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1, groups=2)
        self.bn2d_1 = nn.BatchNorm2d(8)
        self.leak_relu_1 = nn.LeakyReLU(0.3)
        self.con2d_2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, groups=8)
        self.bn2d_2 = nn.BatchNorm2d(16)
        self.leak_relu_2 = nn.LeakyReLU(0.3)
        self.con2d_3 = nn.Conv2d(in_channels=16, out_channels=out_channels, kernel_size=3, padding=1, groups=2)
        self.bn2d_3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.con2d_1(x)
        x = self.bn2d_1(x)
        x = self.leak_relu_1(x)
        x = self.con2d_2(x)
        x = self.bn2d_2(x)
        x = self.leak_relu_2(x)
        x = self.con2d_3(x)
        output = self.bn2d_3(x)
        return output


if __name__ == '__main__':
    pass
