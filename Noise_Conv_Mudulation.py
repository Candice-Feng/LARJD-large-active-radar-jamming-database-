import copy
import numpy
import numpy.fft as fft
from matplotlib import pyplot as plt
from lfm import SignalLinearFrequency
from Radio_Noise import JammingDirectNoise
from scipy.signal import convolve
from scipy.signal import hilbert
from scipy import interpolate
import pandas as pd
import random
import pickle
from filter import get_pass_filter

class JammingNoiseConv:
    def para_init(self, origin_signal, f_trans=0):
        self.origin_signal = copy.copy(origin_signal)
        f_s = self.origin_signal.f_s
        band = self.origin_signal.band
        t_duration = self.origin_signal.t_duration
        t_0 = self.origin_signal.t_0
        f_0 = self.origin_signal.f_0


        self.weightnoise = numpy.random.normal(size=self.origin_signal.t_range.shape)
        self.jamming = JammingDirectNoise().para_init(f_0=f_0, band=band, t_duration=t_duration, f_s=f_s, t_0=t_0)

        self.output_f = fft.fftshift(fft.fft(self.weightnoise))
        filter = get_pass_filter(self.origin_signal.t_range.shape[0], self.origin_signal.f_s, t_0-band/2, t_0+band/2)
        self.weightnoisef = self.output_f * filter
        self.weightnoise = fft.ifft(fft.ifftshift(self.weightnoisef))
        self.jamming.time_domain = hilbert_transform_complex(self.weightnoise)
        # self.jamming = JammingDirectNoise().para_init(f_0=band / 2, band=band, t_duration=t_duration,f_s=f_s, t_0=t_0)
        output = convolve( self.origin_signal.time_domain,self.jamming.time_domain)
        self.t_range = numpy.arange(0, len(output))
        self.f_range = numpy.linspace(-f_s / 2, f_s / 2 - f_s / self.t_range.shape[0], self.t_range.shape[0])
        self.time_domain = output
        return self



def hilbert_transform_complex(signal):
    analytic_signal = numpy.fft.fft(signal)
    n = len(signal)

    if n % 2 == 0:
        analytic_signal[int(n / 2) + 1:] = 0
    else:
        analytic_signal[(n + 1) // 2:] = 0

    # 保留复数形式，将原始信号与Hilbert变换的虚部相加
    hilbert_result = numpy.fft.ifft(analytic_signal)

    return hilbert_result

if __name__ == '__main__':
    num_signals = 10
    new_data_points = 2000  # 新的数据点数
    generated_signals = []
    combined_array = numpy.empty((0, 2001), dtype=complex)
    for _ in range(num_signals):

      origin_signal = SignalLinearFrequency().para_init(f_0=50e6, band=15e6, t_duration=5e-5, f_s=40e6, t_0=-2.5e-5)
      signal = JammingNoiseConv().para_init(origin_signal=origin_signal)
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
      label = 5
      modified_array = interpolated_signal.copy()
      modified_array = numpy.insert(modified_array, 0, label)
      combined_array = numpy.vstack((combined_array, modified_array))
    # existing_data = pd.read_csv('../Database/train_data_list.tsv', sep='\t', header=None)
    # df_array = numpy.vstack((existing_data, combined_array))
    # # 打乱数组的行序
    # numpy.random.shuffle(df_array)
    # # 将连接后的数组转换为 DataFrame
    # df = pd.DataFrame(df_array)
    # # 将DataFrame保存为.tsv文件
    # df.to_csv('../Database/train_data_list.tsv', sep='\t', index=False, header=False)

    plt.figure(1)
    plt.plot(signal.t_range, signal.time_domain.real)
    plt.figure(5)
    plt.plot(signal.t_range, signal.time_domain.imag)
    # plt.savefig('ncmtdomain.jpg', format='jpg')
    # plt.savefig('ncmtdomain.eps', format='eps', dpi=300)

    signal.f_domain = fft.fftshift(fft.fft(signal.time_domain))
    plt.figure(2)
    plt.plot(signal.f_range, numpy.abs(signal.f_domain.real))
    plt.figure(6)
    plt.plot(signal.f_range, numpy.abs(signal.f_domain.imag))

    # plt.savefig('ncmfdomain.jpg', format='jpg')
    # plt.savefig('ncmfdomain.eps', format='eps', dpi=300)
    power_spectrum = numpy.abs(signal.f_domain) ** 2
    plt.figure(3)
    plt.specgram(signal.time_domain, Fs=origin_signal.f_s)
    plt.title('Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Intensity [dB]')
    # plt.savefig('ncmtfdomain.jpg', format='jpg')
    # plt.savefig('ncmtfdomain.eps', format='eps', dpi=300)

    plt.show()

