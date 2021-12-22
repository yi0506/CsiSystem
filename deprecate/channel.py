"""
@description 两个信道模型
@date 2021/10/20
@author leeyj
"""
import math
import numpy as np
import scipy.special as special
from matplotlib import pyplot as plt


def mimo_channel(nt, nr, nc, nray, n=72, wideband=False):
    """
    description: 生成毫米波信道聚类模型，默认采用单用户和ULA的天线模型
    reference: 《Spatially sparse precoding in millimeter wave MIMO systems》
    input:
        nt: 发射天线数
        nr: 接收天线数
        nc: 簇的个数
        nray: 每个簇的路径数
        n: 如果为宽带，则n代表OFDM的子载波数
    output:
        H: 最后生成的信道矩阵H
        H_F: 频域信道矩阵
    """
    E_aoa = 2 * math.pi * np.random.rand(nc, 1)
    sigma_aoa = 10 * math.pi / 180
    b_aoa = sigma_aoa / math.sqrt(2)
    u = np.random.rand(nc, nray) - 0.5
    aoa = E_aoa.repeat(nray, axis=1) - b_aoa * np.sign(u) * np.log(1 - 2 * np.abs(u))
    sin_aoa = np.sin(aoa)

    E_aod = 2 * math.pi * np.random.rand(nc, 1)
    sigma_aod = 10 * math.pi / 180
    b_aod = sigma_aod / math.sqrt(2)
    aod = E_aod.repeat(nray, axis=1) - b_aod * np.sign(u) * np.log(1 - 2 * np.abs(u))
    sin_aod = np.sin(aod)

    H_ray = np.zeros((nr, nt, nc, nray)) * complex(1, 1)
    H_rayleigh = complex(np.random.randn(1), np.random.randn(1)) / np.sqrt(2)
    nt_angle = np.arange(nt).reshape(nt, 1)*math.pi*complex(0, 1)
    nr_angle = np.arange(nr).reshape(nr, 1)*math.pi*complex(0, 1)

    for i in range(nc):
        for j in range(nray):
            H_ray[:, :, i, j] = H_rayleigh*np.exp(nr_angle*sin_aod[i, j]).dot(np.transpose(np.exp(nt_angle*sin_aoa[i, j])))/np.sqrt(nr*nt)

    H_nc = np.sum(H_ray, 3)
    sin_aoa = np.reshape(sin_aoa, (1, sin_aoa.size), order='F')
    sin_aod = np.reshape(sin_aod, (1, sin_aod.size), order='F')
    A = np.kron(sin_aoa, nt_angle)
    B = np.kron(sin_aod, nr_angle)
    AT = 1 / np.sqrt(nt) * np.exp(A)
    AR = 1 / np.sqrt(nr) * np.exp(B)
    # 窄带信道
    H = H_rayleigh*AR.dot(AT.conj().T)

    # 宽带信道
    if wideband:
        H_f = np.zeros(shape=(nr, nt, n))*complex(1, 1)
        H_nc_buffer = np.zeros((nr, nt, nc))*complex(1, 1)
        for i in range(n):
            for j in range(nc):
                H_nc_buffer[:, :, j] = H_nc[:, :, j]*np.exp(-1j*2*math.pi*i*j/n)
            H_f[:, :, i] = np.sum(H_nc_buffer, 2)
        return H_f
    return H


def gen_channel(nt, nr, nc, nray):
    """
    description: 生成毫米波信道聚类模型，默认采用单用户和ULA的天线模型
    reference: 《Spatially sparse precoding in millimeter wave MIMO systems》
    input:
        nt: 发射天线数
        nr: 接收天线数
        nc: 簇的个数
        nray: 每个簇的路径数
    output:
        H: 最后生成的信道矩阵H
    parameter:
       d: 归一化天线间隔
       sig_gain: 每个簇增益的标准差, norm(sigGain,2)^2 = Nt*Nr/Nray, 归一化为E[|H|^2] = Nt*Nr;
       sig_angle: 离开方位角、离开仰角、到达方位角和到达仰角的角扩展
       dir_tx: TX天线定向参数 [phi_min, phi_max][theta_min, theta_max]
       dir_rx: RX天线定向参数 [phi_min, phi_max][theta_min, theta_max]
       phi: 方向角
       theta: 仰角
    """
    # 天线的定向参数，可以进行改动，如果要针对多用户，就需要吧dir_tx和dir_rx提出
    dir_tx = np.array([-60, 60]) * np.pi / 180
    dir_rx = np.array([-180, 180]) * np.pi / 180
    d = 0.5  # 归一化天线间隔
    sig_gain = np.sqrt((nt * nr) / (nc * nray)) * np.ones(shape=(nc, 1))
    sig_angle = 7.5 / 180 * math.pi * np.ones(shape=(2, 1))  # 角度扩展

    nps = nc * nray  # 总的路径数
    gain_alpha = np.zeros(shape=(nps, 1)) * complex(1, 1)  # 路径的复增益

    # 平均角均匀分布
    mean_angle = np.zeros(shape=(nc, 2))
    mean_angle[:, 0] = (2 * np.random.rand(nc) - 1) * (dir_tx[1] - dir_tx[0]) / 2 + (dir_tx[1] + dir_tx[0]) / 2
    mean_angle[:, 1] = (2 * np.random.rand(nc) - 1) * (dir_rx[1] - dir_rx[0]) / 2 + (dir_rx[1] + dir_rx[0]) / 2

    at = np.zeros(shape=(nt, nps))*complex(1, 1)
    ar = np.zeros(shape=(nr, nps))*complex(1, 1)

    for i in range(nps):
        icls = np.floor(i / nray)  # 应该是向下取整得到0,1,2,...nc-1
        sig = sig_gain[int(icls)]
        gain_alpha[i] = sig * (np.random.randn(1) + 1j * np.random.randn(1)) / np.sqrt(2)

        mean_depart = mean_angle[int(icls), 0]
        mean_arrival = mean_angle[int(icls), 1]

        sig_depart = sig_angle[0]
        sig_arrival = sig_angle[1]

        ang_d = gen_laplacian(mean_depart, sig_depart)
        ang_a = gen_laplacian(mean_arrival, sig_arrival)

        dir_gain = int(ang_d > dir_tx[0]) * int(ang_d < dir_tx[1]) * int(ang_a > dir_rx[0]) * int(ang_a < dir_rx[1])
        if dir_gain == 1:
            attmp = np.exp(1j*2*np.pi*d*np.sin(ang_d)*np.arange(nt))/np.sqrt(nt)
            at[:, i] = attmp
            artmp = np.exp(1j*2*np.pi*d*np.sin(ang_a)*np.arange(nr))/np.sqrt(nr)
            ar[:, i] = artmp
    return ar.dot(np.diag(gain_alpha.reshape(nps,))).dot(at.conj().T)


def gen_laplacian(mu, sig):
    """
    description: 用参数{mu, sig}生成拉普拉斯变量
    input:
        mu: 均值
        sig: 标准差
    output:
        x: 拉普拉斯变量
    """
    b = sig / np.sqrt(2)
    u = np.random.rand(1) - 1 / 2
    x = mu - b * np.sign(u) * np.log(1 - 2 * np.abs(u)) / np.log(np.exp(1))
    return x


def random_channel(nr, nt):
    """
    description: 一个服从高斯随机分布的信道
    date: 2021/11/9
    input:
        nr: 接收天线
        nt: 发射天线
    output:
        h: 信道矩阵
    """
    # 产生一个0均值和单位方差的高斯随机矩阵
    hiid = np.random.randn(nr, nt)
    ar = np.zeros(shape=(nr, nr))*complex(1, 1)
    at = np.zeros(shape=(nt, nt))*complex(1, 1)
    for i in range(nr):
        for j in range(nr):
            ar[i, j] = special.jv(0, math.pi*abs(i-j))
    for i in range(nt):
        for j in range(nt):
            at[i, j] = special.jv(0, math.pi*abs(i-j))
    return (1/np.trace(ar))*np.sqrt(ar).dot(hiid).dot(np.ssqrt(at))


if __name__ == '__main__':
    h = mimo_channel(32, 1, 32, 2)
    # h = np.reshape(h, (32, 72)).T
    # h_fft_1 = np.fft.fft(h, axis=0)/32
    # h_fft_2 = np.fft.fft(h_fft_1, axis=1)/72
    # h_samp = np.r_[h_fft_2[0][None, :], h_fft_2[41:]]
    # h_samp_real = h_samp.real
    # h_samp_imag = h_samp.imag
    # h_samp_real_imag = np.r_[h_samp_real.flatten(order='F'), h_samp_imag.flatten(order='F')] + 0.5
    # plt.imshow(h_samp_real, cmap='bone')
    # plt.show()
    # plt.imshow(h_samp_imag, cmap='bone')
    # plt.show()
    # plt.stem(h_samp_real_imag)
    # plt.show()
    print(h.shape)

