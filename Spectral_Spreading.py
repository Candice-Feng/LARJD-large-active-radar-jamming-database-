import copy
import numpy
import numpy.fft as fft
from matplotlib import pyplot as plt
from linear_frequency_signal import SignalLinearFrequency
import pandas as pd
import random
class Spectralspread:
    def para_init(self, origin_signal):
        self.origin_signal = copy.copy(origin_signal)
        f_s = self.origin_signal.f_s
        band = self.origin_signal.band
        t_duration = self.origin_signal.t_duration
        t_0 = self.origin_signal.t_0
        f_domain = fft.fftshift(fft.fft(self.origin_signal.time_domain))
        subpulsenum = 50
        t = self.origin_signal.t_range
        frequencemodu = numpy.exp(1j*2*numpy.pi*band*5e2*t**2)
        self.interfered_signal = frequencemodu*self.origin_signal.time_domain


        return self


if __name__ == '__main__':
    num_signals = 10
    new_data_points = 2000  # 新的数据点数
    combined_array = numpy.empty((0, 2001), dtype=complex)
    for _ in range(num_signals):

        origin_signal = SignalLinearFrequency().para_init(f_0=50e6, band=15e6, t_duration=5e-5, f_s=40e6, t_0=-2.5e-5)
        signal = Spectralspread().para_init(origin_signal=origin_signal)
        JNR = random.randint(10, 100)
        max_amplitude = numpy.max(numpy.abs(signal.interfered_signal))
        signal.interfered_signal = signal.interfered_signal + 1 / (
                    10 ** (JNR / 10)) * max_amplitude * numpy.random.normal(0, 1, len(signal.origin_signal.t_range))
        label = 8
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
    # plt.savefig('smstdomain.jpg', format='jpg')
    # plt.savefig('smstdomain.eps', format='eps', dpi=300)

    signal.f_domain = fft.fftshift(fft.fft(signal.interfered_signal))
    plt.figure(2)
    plt.plot(signal.origin_signal.f_range, numpy.abs(signal.f_domain))
    # plt.savefig('smsfdomain.jpg', format='jpg')
    # plt.savefig('smsfdomain.eps', format='eps', dpi=300)
    power_spectrum = numpy.abs(signal.f_domain) ** 2
    plt.figure(3)
    plt.specgram(signal.interfered_signal, Fs=origin_signal.f_s)
    plt.title('Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Intensity [dB]')
    # plt.savefig('smstfdomain.jpg', format='jpg')
    # plt.savefig('smstfdomain.eps', format='eps', dpi=300)

    plt.show()

