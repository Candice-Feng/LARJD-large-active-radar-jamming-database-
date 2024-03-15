import numpy as np
import numpy
import numpy.fft as fft
from matplotlib import pyplot as plt
from base_signal import BaseSignal
from filter import get_pass_filter
from scipy import interpolate
import pandas as pd
import random

class JammingDirectNoise(BaseSignal):
    def get_signal(self):
        # 设定期望和方差
        mean_value = 0.0  # 期望（均值）
        variance_value = 1  # 方差
        sigma = self.t_duration/1e3  # 高斯函数的标准差，用于控制脉宽
        noise = (numpy.random.normal(loc=mean_value, scale=numpy.sqrt(variance_value), size=self.t_range.shape) +
                 1j * numpy.random.normal(loc=mean_value, scale=numpy.sqrt(variance_value), size=self.t_range.shape))
        # rect_function = numpy.where((-1.5e-5 <= self.t_range) & (self.t_range <= 1.5e-5), 1, 0)

        # 将原始信号乘以矩形函数
        noise = noise

        noise_f = fft.fftshift(fft.fft(noise))
        filter = get_pass_filter(self.t_range.shape[0], self.f_s, self.t_0-self.band/2, self.t_0+self.band/2  )
        output = 2*noise_f * filter
        radio_signal = fft.ifft(fft.ifftshift(output))
        # 生成多个缩短脉宽的高斯脉冲干扰并叠加（纵轴上翻折）
        periodic_gaussian_pulse = np.zeros_like(self.t_range)
        segment_duration = self.t_duration / num_pulses
        t = np.linspace(0, self.t_duration, new_data_points, endpoint=False)

        for i in range(num_pulses):
            pulse_center = (i + 0.5) * segment_duration  # 将每个脉冲放在每个duration/n的中间位置
            gaussian_pulse = np.exp(-(t- pulse_center) ** 2 / (2 * sigma ** 2))
            periodic_gaussian_pulse +=  gaussian_pulse  # 纵轴上翻折
        guassion = periodic_gaussian_pulse*radio_signal
        return guassion
    # def get_signal(self):
    #     # 参数设置
    #     T = self.t_duration/10    # 干扰的脉冲周期
    #     tau = T / 2 # 脉冲持续时间
    #     t_step = T / 10
    #     # 生成高斯脉冲干扰信号
    #     t = self.t_range
    #     gaussian_pulse = np.exp(-(t / tau) ** 2 / 2)
    #     pgpj_signal = np.zeros_like(t)
    #
    #     # 将高斯脉冲叠加在周期性上
    #     for i in range(int(self.t_duration / T)):
    #         pgpj_signal += np.roll(gaussian_pulse, i * int(T / t_step))
    #
    #     # 将结果转换到频域
    #     pgpj_signal_f = fft.fftshift(fft.fft(pgpj_signal))
    #
    #     # 使用带通滤波器获取特定频率范围内的信号
    #     filter = get_pass_filter(self.t_range.shape[0], self.f_s, self.t_0 - self.band / 2, self.t_0 + self.band / 2)
    #     pgpj_signal_f_filtered = 2 * pgpj_signal_f * filter
    #
    #     # 将结果转换回时域
    #     pgpj_signal_filtered = fft.ifft(fft.ifftshift(pgpj_signal_f_filtered))
    #
    #     return pgpj_signal_filtered

if __name__ == '__main__':
    num_signals = 10
    new_data_points = 2000  # 新的数据点数
    # 参数设置
    num_pulses = numpy.random.randint(10, 20)  # 高斯脉冲的数量
    combined_array = numpy.empty((0, 2001), dtype=complex)

    for _ in range(num_signals):
        signal = JammingDirectNoise().para_init(f_0=50e6, band=15e6, t_duration=5e-5, f_s=40e6, t_0=-2.5e-5)
        JNR = random.randint(10, 100)
        max_amplitude = numpy.max(numpy.abs(signal.time_domain))
        signal.time_domain = signal.time_domain + 1 / (10 ** (JNR / 10)) * max_amplitude * numpy.random.normal(
            0, 1, len(signal.t_range))
        label = 7
        modified_array = signal.time_domain.copy()
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
    plt.plot(signal.t_range, signal.time_domain.real)
    # plt.savefig('pgptdomain.jpg', format='jpg')
    # plt.savefig('pgptdomain.eps', format='eps', dpi=300)

    signal.f_domain = fft.fftshift(fft.fft(signal.time_domain))
    plt.figure(2)
    plt.plot(signal.f_range, numpy.abs(signal.f_domain))
    # plt.savefig('pgpfdomain.jpg', format='jpg')
    # plt.savefig('pgpfdomain.eps', format='eps', dpi=300)

    power_spectrum = numpy.abs(signal.f_domain) ** 2
    plt.figure(3)
    plt.specgram(signal.time_domain, Fs=signal.f_s)
    plt.title('Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Intensity [dB]')
    # plt.savefig('pgptfdomain.jpg', format='jpg')
    # plt.savefig('pgptfdomain.eps', format='eps', dpi=300)

    plt.show()



