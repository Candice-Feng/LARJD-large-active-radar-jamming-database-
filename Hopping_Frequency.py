import copy
import numpy
import numpy.fft as fft
from matplotlib import pyplot as plt
from linear_frequency_signal import SignalLinearFrequency
import pandas as pd
import random
import pickle
class Spectralspread:
    def para_init(self, origin_signal):
        self.origin_signal = copy.copy(origin_signal)
        f_s = self.origin_signal.f_s
        band = self.origin_signal.band
        t_duration = self.origin_signal.t_duration
        t_0 = self.origin_signal.t_0

        # 原始信号频谱
        f_domain = fft.fftshift(fft.fft(self.origin_signal.time_domain))

        # 生成跳频干扰
        N = numpy.random.randint(5,16)  # 幅度序列、频率序列、相位序列的数量
        A_n = numpy.random.rand(N)
        f_n = numpy.random.rand(N) * f_s  # 频率范围可以根据需要进行调整
        j_n = numpy.random.rand(N) * 2 * numpy.pi
        T_H = t_duration/N  # 跳频持续时间

        hopping_jamming = self.hopping_frequency_jamming(self.origin_signal.t_range, A_n, f_n, j_n, T_H, t_0)

        # 将跳频干扰加入原始信号
        self.interfered_signal =  hopping_jamming*self.origin_signal.time_domain

        return self

    def hopping_frequency_jamming(self, t, A_n, f_n, j_n, T_H, t_p):
        jamming_signal = numpy.zeros_like(t, dtype=numpy.complex128)
        for n in range(len(A_n)):
            pulse = numpy.where((t >= n * T_H) & (t < (n + 1) * T_H), 1, 0)
            jamming_signal += 1 * numpy.exp(1j * (2 * numpy.pi * f_n[n] * t + j_n[n])) * pulse

        return jamming_signal


if __name__ == '__main__':
    num_signals = 10
    new_data_points = 2000  # 新的数据点数
    combined_array = numpy.empty((0, 2001), dtype=complex)
    for _ in range(num_signals):

      origin_signal = SignalLinearFrequency().para_init(f_0=50e6, band=15e6, t_duration=5e-5, f_s=40e6, t_0=0)
      signal = Spectralspread().para_init(origin_signal=origin_signal)
      JNR = random.randint(10, 100)
      max_amplitude = numpy.max(numpy.abs(signal.interfered_signal))
      signal.interfered_signal = signal.interfered_signal + 1 / (10 ** (JNR / 10)) * max_amplitude * numpy.random.normal(0,1,len(signal.origin_signal.t_range))
      label = 10
      modified_array = signal.interfered_signal.copy()
      modified_array = numpy.insert(modified_array, 0, label)
      combined_array = numpy.vstack((combined_array, modified_array))
    existing_data = pd.read_csv('../Database/train_data_list.tsv', sep='\t', header=None)
    df_array = numpy.vstack((existing_data, combined_array))
    # 打乱数组的行序
    numpy.random.shuffle(df_array)
    # 将连接后的数组转换为 DataFrame
    df = pd.DataFrame(df_array)
    # 将DataFrame保存为.tsv文件
    df.to_csv('../Database/train_data_list.tsv', sep='\t', index=False, header=False)

    plt.figure(1)
    plt.plot(signal.origin_signal.t_range, signal.interfered_signal.real)
    # plt.savefig('hftdomain.jpg', format='jpg')
    # plt.savefig('hftdomain.eps', format='eps', dpi=300)

    signal.f_domain = fft.fftshift(fft.fft(signal.interfered_signal))
    plt.figure(2)
    plt.plot(signal.origin_signal.f_range, numpy.abs(signal.f_domain))
    # plt.savefig('hffdomain.jpg', format='jpg')
    # plt.savefig('hffdomain.eps', format='eps', dpi=300)
    power_spectrum = numpy.abs(signal.f_domain) ** 2
    plt.figure(3)
    plt.specgram(signal.interfered_signal, Fs=origin_signal.f_s)
    plt.title('Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Intensity [dB]')
    # plt.savefig('hftfdomain.jpg', format='jpg')
    # plt.savefig('hftfdomain.eps', format='eps', dpi=300)

    plt.show()

