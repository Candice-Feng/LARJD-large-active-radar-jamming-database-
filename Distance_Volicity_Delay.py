import copy
import numpy
import numpy.fft as fft
from matplotlib import pyplot as plt
from linear_frequency_signal import SignalLinearFrequency
import random
import pandas as pd
class Distancedelay:
    def para_init(self, origin_signal):
        self.origin_signal = copy.copy(origin_signal)
        f_s = self.origin_signal.f_s
        band = self.origin_signal.band
        t_duration = self.origin_signal.t_duration
        t_0 = self.origin_signal.t_0

        delay_time = t_duration/random.randint(5,10)
        self.interfered_signal = numpy.zeros_like(self.origin_signal.time_domain)
        delay_index = int(delay_time * f_s)
        sample_index = int(t_duration/5 * f_s)
        start = int(self.origin_signal.t_0 * f_s)
        # Identify indices corresponding to the time window
        start_index = start
        end_index = start_index+int(t_duration/2 * f_s)
        start_direct_index = start_index+delay_index
        end_direct_index = end_index + delay_index

        # Perform sampling
        sampled_signal = 1.5*self.origin_signal.time_domain[start_index:end_index]

        # Directly forward the sampled signal
        self.interfered_signal[start_direct_index:end_direct_index] = sampled_signal


        f_domain = fft.fftshift(fft.fft(self.interfered_signal))
        new_f_domian = 0.1*random.randint(10,20) * f_domain
        delay_time = t_duration / random.randint(5,10)
        self.interfered_signal = numpy.zeros_like(f_domain)
        delay_index = int(delay_time * f_s)
        sample_index = int(t_duration / 5 * f_s)
        start = int(self.origin_signal.t_0 * f_s)
        # Identify indices corresponding to the time window
        start_index = start
        end_index = start_index + int(t_duration * f_s) - int(t_duration / 5 * f_s)
        start_direct_index = start_index + delay_index
        end_direct_index = end_index + delay_index

        # Perform sampling
        sampled_signal = new_f_domian[start_index:end_index]

        # Directly forward the sampled signal
        self.interfered_signal[start_direct_index:end_direct_index] = sampled_signal
        self.interfered_signal = fft.ifft(fft.ifftshift(self.interfered_signal))

        return self


if __name__ == '__main__':
    num_signals = 10
    new_data_points = 2000  # 新的数据点数
    combined_array = numpy.empty((0, 2001), dtype=complex)
    for _ in range(num_signals):
        origin_signal = SignalLinearFrequency().para_init(f_0=50e6, band=15e6, t_duration=5e-5, f_s=40e6, t_0=0)
        signal = Distancedelay().para_init(origin_signal=origin_signal)
        JNR = random.randint(10, 100)
        max_amplitude = numpy.max(numpy.abs(signal.interfered_signal))
        signal.interfered_signal = signal.interfered_signal + 1 / (
                    10 ** (JNR / 10)) * max_amplitude * numpy.random.normal(0, 1, len(signal.origin_signal.t_range))
        label = 18
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
    # plt.savefig('dvdtdomain.jpg', format='jpg')
    # plt.savefig('dvdtdomain.eps', format='eps', dpi=300)

    signal.f_domain = fft.fftshift(fft.fft(signal.interfered_signal))
    plt.figure(2)
    plt.plot(signal.origin_signal.f_range, numpy.abs(signal.f_domain))
    # plt.savefig('dvdfdomain.jpg', format='jpg')
    # plt.savefig('dvdfdomain.eps', format='eps', dpi=300)
    power_spectrum = numpy.abs(signal.f_domain) ** 2
    plt.figure(3)
    plt.specgram(signal.interfered_signal, Fs=origin_signal.f_s)
    plt.title('Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Intensity [dB]')
    # plt.savefig('dvdtfdomain.jpg', format='jpg')
    # plt.savefig('dvdtfdomain.eps', format='eps', dpi=300)
    plt.show()

