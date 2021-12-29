import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


from utils import gs_noise

class ISTANetplusConfiguration(object):
    """ISTANetplus配置"""
    layer_num = 9
    data_length = 2048
    maxtrix_len = 32  # 信号矩阵为方阵，其长度
    channel_num = 2  # 矩阵通道数
    kerner_size = 3
    stride = 1
    padding = 1
    network_name = "ISTANet+"  # 网络名称


class ISTANetplus(torch.nn.Module):
    
    def __init__(self, LayerNo):
        super(ISTANetplus, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for _ in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Phix, Phi, Qinit, snr=None):
        """
        Phix 是 Phi * x = y 观测向量，[batch, 2048/ratio] --> [batch, m]
        Phi 是观测矩阵 [2048/ratio, 2048]
        Qinit ||QY-X||的最小二乘解，Q是线性映射矩阵，[2048, m]
        """
        # 加入噪声
        Phix = gs_noise(Phix, snr)
        # (2048, m) * (m, 2048) = (2048, 2048)
        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        # (batch, m) * (m, 2048) = (batch, 2048)
        PhiTb = torch.mm(Phix, Phi)
        # x_0 = y * Qinit.T   (batch, m) * (m, 2048) = (batch, 2048)
        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))
        layers_sym = []   # for computing symmetric loss

        # 每一个phase进行迭代计算
        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[i](x, PhiTPhi, PhiTb)
            layers_sym.append(layer_sym)
        # 取最后一次输出作为最终结果
        x_final = x

        return [x_final, layers_sym]


# Define ISTA-Net-plus Block
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))  # 迭代步长
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))  # 阈值函数步长

        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 2, 3, 3)))  # 32个，2通道，3*3卷积核

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))

        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(2, 32, 3, 3)))

    def forward(self, x, PhiTPhi, PhiTb):
        # r_k (batch, 2, 32, 32)
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        x_input = x.view(-1, 2, 32, 32) 

        # D(·) (batch, 32, 32, 32) 
        x_D = F.conv2d(x_input, self.conv_D, padding=1)
        
        # F(·) (batch, 32, 32, 32) 
        x = F.conv2d(x_D, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)  
        
        # soft(·) (batch, 32, 32, 32)
        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        # x_k = F~(·)  (batch, 32, 32, 32)
        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)

        # G(·)  (batch, 2, 32, 32)
        x_G = F.conv2d(x_backward, self.conv_G, padding=1)

        # 残差
        x_pred = x_input + x_G
        
        # 预测值
        x_pred = x_pred.view(-1, 2048)
        
        # 恒等约束
        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_D_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_D_est - x_D

        return [x_pred, symloss]