import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from base_signal import BaseSignal
from scipy.signal.windows import boxcar
import pandas as pd
from scipy import interpolate
import random
class SignalLinearFrequency(BaseSignal):
    def get_signal(self):
        k = self.band / self.t_duration
        output = np.exp(1j * 2 * np.pi * ((self.f_0 - self.band / 2) * self.t_range + k * self.t_range ** 2 / 2))
        # # 生成矩形函数
        # rect_function = boxcar( self.t_range / 2e-5)
        rect_function = np.where((-1e-5 <= self.t_range) & (self.t_range <= 1e-5), 1, 0)

        # 将原始信号乘以矩形函数
        output = output
        return output


if __name__ == '__main__':
    num_signals = 2000  # 生成100个LFM信号
    new_data_points = 2000  # 新的数据点数

    for _ in range(num_signals):
        signal = SignalLinearFrequency().para_init(f_0=5e6/2, band=5e6, t_duration=5e-5, f_s=40e6, t_0=-2.5e-5)


    plt.figure(1)
    plt.plot(signal.t_range, signal.time_domain)

    signal.f_domain = fft.fftshift(fft.fft(signal.time_domain))
    plt.figure(2)
    plt.plot(signal.f_range, np.abs(signal.f_domain))
    plt.figure(3)
    plt.specgram(signal.time_domain, Fs=signal.f_s)
    plt.title('Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Intensity [dB]')

    out = signal.match_filtering(signal)
    plt.figure(4)
    plt.plot(out.t_range, out.time_domain)

    plt.show()