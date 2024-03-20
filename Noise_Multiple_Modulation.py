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
        self.amplitude = kwargs['amplitude']
        super().para_init(f_0, band, t_duration, f_s, t_0, with_window=False)
        return self

    def get_signal(self):
        f_s = self.f_s
        band = self.band
        t_duration = self.t_duration
        t_0 = self.t_0
        f_0 = self.f_0
        mean_value = 0.0  # 期望（均值）
        variance_value = 1  # 方差
        output = (numpy.random.normal(loc=mean_value, scale=numpy.sqrt(variance_value), size=self.t_range.shape))

        # output = numpy.random.normal(size=self.t_range.shape)
        self.jamming = JammingDirectNoise().para_init(f_0=f_0, band=band, t_duration=t_duration, f_s=f_s, t_0=t_0)

        output = self.amplitude * numpy.cos(2 * numpy.pi *output)

        output_f = fft.fftshift(fft.fft(output))
        filter = get_pass_filter(self.t_range.shape[0], self.f_s, self.t_0-self.band/2, self.t_0+self.band/2)
        output_f = output_f * filter
        return fft.ifft(fft.ifftshift(output_f))


class JammingNoiseMulti:
    def para_init(self, origin_signal, amplitude):
        self.origin_signal = origin_signal

        noise = CosNoise().para_init(f_0=self.origin_signal.f_0, band=self.origin_signal.band,
                                     t_duration=self.origin_signal.t_duration, f_s=self.origin_signal.f_s,
                                     t_0=self.origin_signal.t_0, amplitude=amplitude)

        self.t_range = origin_signal.t_range
        self.f_range = origin_signal.f_range
        self.time_domain =self.origin_signal.time_domain * noise.time_domain
        return self


if __name__ == '__main__':
    num_signals = 10
    new_data_points = 2000  # 新的数据点数
    generated_signals = []
    combined_array = numpy.empty((0, 2001), dtype=complex)
    for _ in range(num_signals):
        origin_signal = SignalLinearFrequency().para_init(f_0=50e6, band=15e6, t_duration=5e-5, f_s=40e6, t_0=-2.5e-5)
        signal = JammingNoiseMulti().para_init(origin_signal=origin_signal, amplitude= random.random())
        JNR = random.randint(10, 100)
        max_amplitude = numpy.max(numpy.abs(signal.time_domain))
        signal.time_domain = signal.time_domain + 1 / (10 ** (JNR / 10)) * max_amplitude * numpy.random.normal(
            0, 1, len(signal.t_range))
        x_old = numpy.linspace(0, signal.t_range[-1], len(signal.time_domain))
        x_new = numpy.linspace(0, signal.t_range[-1], new_data_points)
        interpolator_real = interpolate.interp1d(x_old, signal.time_domain.real, kind='cubic')
        interpolated_real = interpolator_real(x_new)

        # 对虚部进行插值
        interpolator_imag = interpolate.interp1d(x_old, signal.time_domain.imag, kind='cubic')
        interpolated_imag = interpolator_imag(x_new)
        # 合并实部和虚部得到复数形式的插值信号
        interpolated_signal = interpolated_real + 1j * interpolated_imag
        label = 6
        modified_array = interpolated_signal.copy()
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
    plt.plot(x_new, interpolated_signal.real)
    # plt.savefig('nmmtdomain.jpg', format='jpg')
    # plt.savefig('nmmtdomain.eps', format='eps', dpi=300)

    signal.f_domain = fft.fftshift(fft.fft(signal.time_domain))
    plt.figure(2)
    plt.plot(signal.f_range, numpy.abs(signal.f_domain))
    # plt.savefig('nmmfdomain.jpg', format='jpg')
    # plt.savefig('nmmfdomain.eps', format='eps', dpi=300)
    power_spectrum = numpy.abs(signal.f_domain) ** 2
    plt.figure(3)
    plt.specgram(signal.time_domain, Fs=origin_signal.f_s)
    plt.title('Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Intensity [dB]')
    # plt.savefig('nmmtfdomain.jpg', format='jpg')
    # plt.savefig('nmmtfdomain.eps', format='eps', dpi=300)

    plt.show()