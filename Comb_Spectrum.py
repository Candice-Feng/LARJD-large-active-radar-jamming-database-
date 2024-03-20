import numpy
import numpy.fft as fft
from matplotlib import pyplot as plt
from base_signal import BaseSignal
from Radio_Noise import JammingDirectNoise
from scipy import interpolate
import random
import pandas as pd
class CombSpectrumJamming(BaseSignal):
    def para_init(self, f_0, band, t_duration, f_s, t_0, with_window=False, **kwargs):
        self.comb_amplitudes = kwargs['comb_amplitudes']
        self.comb_frequencies = kwargs['comb_frequencies']
        self.comb_phase_shifts = kwargs['comb_phase_shifts']
        self.carrier_amplitude = kwargs['carrier_amplitude']

        super().para_init(f_0, band, t_duration, f_s, t_0, with_window=False)
        return self

    def get_signal(self):
        f_0 = self.f_0
        band = self.band
        t = self.t_range
        U_0 = self.carrier_amplitude
        K_FM = 2 * numpy.pi * band*0.4/ t[-1]

        cj = numpy.zeros_like(t, dtype=numpy.complex128)

        for i in range(len(self.comb_amplitudes)):
            # 使用梳状频谱公式生成每个梳状结构的信号分量
            comb_structure = self.comb_amplitudes[i] * numpy.exp(
                1j * (
                        2 * numpy.pi * self.comb_frequencies[i] * t +
                        2 * numpy.pi * K_FM * t ** 2 +
                        self.comb_phase_shifts[i]
                )
            )

            # 将每个梳状结构的信号叠加到总的梳状频谱干扰信号上
            cj += comb_structure

        # 添加其他噪声或调制信号（如果需要）
        output = JammingDirectNoise().para_init(f_0=f_0, band=band, t_duration=self.t_duration,
                                                f_s=self.f_s, t_0=self.t_0).time_domain
        output = band / 5 * output.real

        return cj  # Exclude the noise component from the output

if __name__ == '__main__':
    num_signals = 10
    new_data_points = 2000  # 新的数据点数
    combined_array = numpy.empty((0, 2001), dtype=complex)
    for _ in range(num_signals):
        signal = CombSpectrumJamming().para_init(
            f_0=50e6, band=15e6, t_duration=5e-5, f_s=40e6, t_0=-2.5e-5,
            comb_amplitudes=numpy.random.rand(2), comb_frequencies=15e6 * numpy.random.rand(2),
            comb_phase_shifts=0.1 * numpy.random.rand(2), carrier_amplitude=0.5
        )
        JNR = random.randint(10, 100)
        max_amplitude = numpy.max(numpy.abs(signal.time_domain))
        signal.time_domain = signal.time_domain + 1 / (10 ** (JNR / 10)) * max_amplitude * numpy.random.normal(
            0, 1, len(signal.t_range))
        label = 9
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
    # plt.savefig('cstdomain.jpg', format='jpg')
    # plt.savefig('cstdomain.eps', format='eps', dpi=300)

    signal.f_domain = fft.fftshift(fft.fft(signal.time_domain))
    plt.figure(2)
    plt.plot(signal.f_range, numpy.abs(signal.f_domain))
    # plt.savefig('csfdomain.jpg', format='jpg')
    # plt.savefig('csfdomain.eps', format='eps', dpi=300)

    power_spectrum = numpy.abs(signal.f_domain) ** 2
    plt.figure(3)
    plt.specgram(signal.time_domain, Fs=signal.f_s)
    plt.title('Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Intensity [dB]')
    # plt.savefig('cstfdomain.jpg', format='jpg')
    # plt.savefig('cstfdomain.eps', format='eps', dpi=300)

    plt.show()
