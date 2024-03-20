import copy
import numpy
import numpy.fft as fft
from matplotlib import pyplot as plt
from linear_frequency_signal import SignalLinearFrequency
import random
import pandas as pd
import pickle


class SlicedLFM:
    def para_init(self, origin_signal, num_slices):
        self.origin_signal = copy.copy(origin_signal)
        f_s = self.origin_signal.f_s
        band = self.origin_signal.band
        t_duration = self.origin_signal.t_duration
        t_0 = self.origin_signal.t_0

        self.num_slices = num_slices
        self.interfered_signal = numpy.zeros_like(self.origin_signal.time_domain)
        # num_slices = 10
        repeat_time = numpy.random.randint(3, 7)

        # Divide the pulse width into slices
        slice_duration = t_duration / num_slices
        slice_t_0 = t_0 + (1 - 1 / num_slices) * (t_duration / 2)
        # Generate sliced and repeated LFM signals
        for i in range(num_slices):
            # rect_function = numpy.where((self.origin_signal.t_0 + i * slice_duration <= self.origin_signal.t_range) & (
            #             self.origin_signal.t_range <= self.origin_signal.t_0 + i * slice_duration +slice_duration/5), 1, 0)
            # sliced_signal = rect_function*self.origin_signal.time_domain
            # sliced_signal_time_domain = sliced_signal
            # Number of slices

            slice = int(slice_duration * f_s)
            start_index = int(t_0 * f_s + i * slice) + 1000
            end_index = int(start_index + slice)
            extract = int(slice / repeat_time)
            extract_start_time = start_index
            extract_end_time = start_index + extract
            sub_signal = self.origin_signal.time_domain[extract_start_time:extract_end_time]
            for j in range(repeat_time):
                start = (start_index + j * extract)
                end = (start_index + (j + 1) * extract)
                self.interfered_signal[start:end] += sub_signal

        return self


if __name__ == '__main__':
    num_signals = 200
    new_data_points = 2000  # 新的数据点数
    combined_array = numpy.empty((0, 2001), dtype=complex)
    for _ in range(num_signals):
        origin_signal = SignalLinearFrequency().para_init(f_0=50e6, band=15e6, t_duration=5e-5, f_s=40e6, t_0=-2.5e-5)
        signal = SlicedLFM().para_init(origin_signal=origin_signal, num_slices=random.randint(4, 11))

# 生成示例时序信号
time_points = signal.origin_signal.t_range
signal =signal.interfered_signal.real


# 计算 Poincaré 映射
poincare_map_x = signal[:-1]
poincare_map_y = signal[1:]

# 绘制 Poincaré 图
plt.scatter(poincare_map_x, poincare_map_y, s=5, c='blue', alpha=0.7)
plt.title('Poincaré Map of the Time Series Signal')
plt.xlabel('Signal at Time t')
plt.ylabel('Signal at Time t+1')
plt.show()
