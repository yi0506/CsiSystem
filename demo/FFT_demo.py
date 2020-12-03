# -*- coding: UTF-8 -*-
"""
@author:wsy123
@file:FFT_demo.py
@time:2020/07/20
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号


def fft_process(x):
    """
    X[k] = $\sum_{n=0}^(N-1)x[n]*exp(-j*2*pi*k*n/N)$

    k: frequency index
    N: length of complex sinusoid in samples
    n: 当前样本

    """
    x_row, x_col = x.shape

    # 保证x为列向量,二维数组
    if x_col > x_row:
        x = x.T
    N = x.shape[0]

    n = np.arange(N).reshape(1, N)
    m = n.T * n / N  # [N, 1] * [1, N] = [N, N]
    S = np.exp(-1j * 2 * np.pi * m)
    x_fft = np.matmul(S, x)
    return x_fft, S


def ifft_process(x_fft, S):
    """取实部，获得原始信号"""
    x_row = x_fft.shape[0]
    S_conj = np.conjugate(S)
    return np.matmul(S_conj.T, x_fft) / x_row


def generate_sine():
    '''
    N : number of samples
    A : amplitude
    fs: sample rate
    f0: frequency
    phi: initial phase
    '''
    # generate signal
    N = 50
    A = 0.8
    fs = 44100
    f0 = 1000
    phi = 0.0
    T = 1 / fs
    n = np.arange(N)
    x = A * np.cos(2 * np.pi * f0 * n * T + phi)
    x = x.reshape(x.shape[0], 1)

    plt.figure()
    plt.plot(x)
    plt.show()
    return x


if __name__ == '__main__':
    y = generate_sine()
    x_1, S = fft_process(y)
    x_3 = ifft_process(x_1, S)

    print(y)
    print("*" * 100)
    print(x_1)
    print(np.complex)
    print(np.real(x_3))
    plt.figure()
    plt.plot(np.real(x_3))
    plt.show()
