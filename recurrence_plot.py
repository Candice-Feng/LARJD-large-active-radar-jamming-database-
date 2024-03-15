import numpy as np
import numpy
import numpy.fft as fft
from matplotlib import pyplot as plt
from base_signal import BaseSignal
from filter import get_pass_filter
from linear_frequency_signal import SignalLinearFrequency
from Radio_Noise import JammingDirectNoise
from utils import signal_conv
from scipy import interpolate
import pandas as pd
import random


class CosNoise(BaseSignal):
    def para_init(self, f_0, band, t_duration, f_s, t_0, with_window=False, **kwargs):
        self.amolitude = kwargs['amolitude']
        super().para_init(f_0, band, t_duration, f_s, t_0, with_window=False)
        return self

    def get_signal(self):
        f_s = self.f_s
        band = self.band
        t_duration = self.t_duration
        t_0 = self.t_0
        f_0 = self.f_0
        output = numpy.random.normal(size=self.t_range.shape)
        self.jamming = JammingDirectNoise().para_init(f_0=f_0, band=band, t_duration=t_duration, f_s=f_s, t_0=t_0)

        output = self.amolitude * numpy.exp(1j * 2 * numpy.pi * output)

        output_f = fft.fftshift(fft.fft(output))
        filter = get_pass_filter(self.t_range.shape[0], self.f_s, self.t_0 - self.band / 2, self.t_0 + self.band / 2)
        output_f = output_f * filter
        return fft.ifft(fft.ifftshift(output_f))


class JammingNoiseMulti:
    def para_init(self, origin_signal, amolitude):
        self.origin_signal = origin_signal

        noise = CosNoise().para_init(f_0=self.origin_signal.f_0, band=self.origin_signal.band,
                                     t_duration=self.origin_signal.t_duration, f_s=self.origin_signal.f_s,
                                     t_0=self.origin_signal.t_0, amolitude=amolitude)

        self.t_range = origin_signal.t_range
        self.f_range = origin_signal.f_range
        self.time_domain = self.origin_signal.time_domain * noise.time_domain
        return self


if __name__ == '__main__':
    num_signals = 200
    new_data_points = 2000  # 新的数据点数
    generated_signals = []
    combined_array = numpy.empty((0, 2001), dtype=complex)
    for _ in range(num_signals):
        origin_signal = SignalLinearFrequency().para_init(f_0=50e6, band=15e6, t_duration=5e-5, f_s=40e6, t_0=-2.5e-5)
        signal = JammingNoiseMulti().para_init(origin_signal=origin_signal, amolitude=random.random())


def calculate_similarity_matrix(signal):
    # 使用欧氏距离计算相似度矩阵
    similarity_matrix = np.abs(np.subtract.outer(signal, signal))
    return similarity_matrix

def generate_recurrence_plot(similarity_matrix, threshold):
    # 使用阈值生成循环图
    recurrence_plot = (similarity_matrix < threshold).astype(int)
    return recurrence_plot

# 生成示例时序信号
time_points = signal.t_range
signal =signal.time_domain.real

# 计算相似度矩阵
similarity_matrix = calculate_similarity_matrix(signal)

# 设置阈值
threshold = 0.1

# 生成循环图
recurrence_plot = generate_recurrence_plot(similarity_matrix, threshold)

# 绘制循环图
plt.imshow(recurrence_plot, cmap='binary', origin='upper', extent=[0, len(signal), 0, len(signal)])

plt.savefig('Recurrence Plot.png')
plt.show()
