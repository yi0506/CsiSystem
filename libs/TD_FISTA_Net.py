import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from utils import gs_noise


class TDFISTANetConfiguration(object):
    """ISTANetplus配置"""
    layer_num = 5
    maxtrix_len = 32  # 信号矩阵为方阵，其长度
    channel_num = 2  # 矩阵通道数
    data_length = 2048  # 信号矩阵变成一维向量后的长度 2048 == 2 * 32 *32
    kerner_size = 3
    stride = 1
    padding = 1
    network_name = "TD-FISTANet"  # 网络名称


class TDFISTANet(torch.nn.Module):
    def __init__(self, LayerNo, ratio):
        super(TDFISTANet, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for _ in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)
        self.td = TDBlock(ratio)  # 时间差分压缩模块
        self.Qinit = nn.Parameter(init.xavier_normal_(torch.Tensor(2048 // ratio, 2048)))

    def forward(self, x, snr=None):
        """
        x 是信号向量，[batch, 2048/ratio] --> [batch, m]

        Qinit ||QY-X||的最小二乘解，Q是线性映射矩阵，[2048, m]
        """
        # 压缩 (2048, m)
        Phix = self.td(x)

        # 是否加入噪声
        Phix = gs_noise(Phix, snr)

        # 信号恢复
        # (2048, m) * (m, 2048) = (2048, 2048)
        PhiTPhi = torch.mm(torch.transpose(self.td.Phi, 0, 1), self.td.Phi)
        # (batch, m) * (m, 2048) = (batch, 2048)
        PhiTb = torch.mm(Phix, self.td.Phi)
        # x_0 = y * Qinit.T   (batch, m) * (m, 2048) = (batch, 2048)
        x = torch.mm(Phix, self.Qinit)
        layers_sym = []  # for computing symmetric loss
        h_iter = []  # 计算迭代损失
        h = x.view(-1, 2, 32, 32)  # h_0 初始化为x
        # 每一个phase进行迭代计算
        for i in range(self.LayerNo):
            [x, h, layer_sym] = self.fcs[i](x, h, PhiTPhi, PhiTb)
            layers_sym.append(layer_sym)
            h_iter.append(h.view(-1, 2048))
        # 取最后一次输出作为最终结果
        x_final = x

        return [x_final, h_iter, layers_sym]


class SoftThreadFunc(nn.Module):
    """软阈值函数"""

    def __init__(self) -> None:
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1)
        self.conv_2 = torch.nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.conv_2(x)
        return x


class TDBlock(torch.nn.Module):
    """时间差分模块"""

    def __init__(self, ratio) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor([0.01]))  # 时间差分系数
        self.Phi = nn.Parameter(init.xavier_normal_(torch.Tensor(2048 // ratio, 2048)))
        self.x_pre = 0

    def forward(self, x):
        x = x - self.gamma * self.x_pre  # 马尔科夫时间差分
        Phix = torch.mm(self.Phi, x.T)  # 压缩  (m, 2048) * (2048, batch) = (m, batch)
        self.x_pre = x.detach()  # 保存上一个时刻的数据
        return Phix.T


class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))  # 梯度迭代步长
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))  # 阈值函数步长
        self.eta_step = nn.Parameter(torch.Tensor([0.01]))  # 加速梯度步长
        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 2, 3, 3)))  # 32个，2通道，3*3卷积核

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))

        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(2, 32, 3, 3)))

        self.soft_thread = SoftThreadFunc()

    def forward(self, y_k, h_k, PhiTPhi, PhiTb):
        """
        预测值是h_k，输入是上一次预测值y_(k-1)和h_(k-1)
        """

        # g_k (batch, 2, 32, 32)
        y_k = y_k - self.lambda_step * torch.mm(y_k, PhiTPhi)
        g_k = y_k + self.lambda_step * PhiTb
        x_input = g_k.view(-1, 2, 32, 32)

        # R(·) (batch, 32, 32, 32)
        x_R = F.conv2d(x_input, self.conv_D, padding=1)

        # S(·) (batch, 32, 32, 32)
        x = F.conv2d(x_R, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        # soft(·) (batch, 32, 32, 32)
        # x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))
        x = self.soft_thread(x)

        # S~(·)  (batch, 32, 32, 32)
        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)

        # D(·)  (batch, 2, 32, 32)
        x_D = F.conv2d(x_backward, self.conv_G, padding=1)

        # 预测值 h_k  (batch, 2, 32, 32)
        h_pred = x_input + x_D

        # y_k  (batch, 2, 32, 32)
        y_pred = h_pred + self.eta_step * (h_pred - h_k)
        h_pred = h_pred.view(-1, 2048)

        # 恒等约束
        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_D_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_D_est - x_R

        return [h_pred, y_pred, symloss]