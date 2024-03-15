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
        lfm_signal = np.exp(1j * 2 * np.pi * ((self.f_0 - self.band / 2) * self.t_range + k * self.t_range ** 2 / 2))

        # 生成矩形函数
        rect_function = np.where((-1e-5 <= self.t_range) & (self.t_range <= 1e-5), 1, 0)

        # 将原始信号乘以矩形函数
        lfm_signal = lfm_signal

        # 添加目标回波信号
        target_echo = self.generate_target_echo()

        # 合并信号
        output = lfm_signal + target_echo
        return output

    def generate_target_echo(self):
        # 生成目标回波信号，这里添加了高斯噪声
        target_k = self.band / self.t_duration
        target_echo = np.exp(1j * 2 * np.pi * ((self.f_0 + 1e6) * self.t_range + target_k * self.t_range ** 2 / 2))

        # 添加高斯噪声
        noise_amplitude = random.random()  # 调整噪声的振幅
        noise = noise_amplitude * (np.random.randn(len(self.t_range)) + 1j * np.random.randn(len(self.t_range)))

        target_echo_with_noise = noise

        return target_echo_with_noise


if __name__ == '__main__':
    num_signals = 10  # 生成100个LFM信号
    new_data_points = 2000  # 新的数据点数
    combined_array = np.empty((0, 2001), dtype=complex)
    for _ in range(num_signals):
        signal= SignalLinearFrequency().para_init(f_0=50e6, band=15e6, t_duration=5e-5, f_s=40e6, t_0=-2.5e-5)
        label = 0
        modified_array = signal.time_domain.copy()
        modified_array = np.insert(modified_array, 0, label)
        combined_array = np.vstack((combined_array, modified_array))
    # 打乱数组的行序
    # np.random.shuffle(combined_array)
    #
    # df = pd.DataFrame(combined_array)
    # # 将DataFrame保存为.tsv文件
    # df.to_csv('../Database/train_data_list.tsv', sep='\t', index=False, header=False)
    plt.figure(1)
    plt.plot(signal.t_range, signal.time_domain.real)
    plt.figure(5)
    plt.plot(signal.t_range, signal.time_domain.imag)

    # 保存为JPEG文件
    # plt.savefig('lfmtdomain.jpg', format='jpg')
    # # 保存为EPS文件
    # plt.savefig('lfmtdomain.eps', format='eps', dpi=300)
    signal.f_domain = fft.fftshift(fft.fft(signal.time_domain))
    plt.figure(2)
    plt.plot(signal.f_range, np.abs(signal.f_domain.real))
    plt.figure(6)
    plt.plot(signal.f_range, np.abs(signal.f_domain.imag))
    # 保存为JPEG文件
    # plt.savefig('lfmfdomain.jpg', format='jpg')
    # # 保存为EPS文件
    # plt.savefig('lfmfdomain.eps', format='eps', dpi=300)

    plt.figure(3)
    plt.specgram(signal.time_domain, Fs=signal.f_s)
    plt.title('Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Intensity [dB]')
    # 保存为JPEG文件
    # plt.savefig('lfmtfdomain.jpg', format='jpg')
    # # 保存为EPS文件
    # plt.savefig('lfmtfdomain.eps', format='eps', dpi=300)

    out = signal.match_filtering(signal)
    plt.figure(4)
    plt.plot(out.t_range, out.time_domain)

    plt.show()
