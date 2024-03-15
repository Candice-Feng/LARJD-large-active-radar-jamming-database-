import copy
import random
import pandas as pd
import numpy
import numpy.fft as fft
from matplotlib import pyplot as plt
from linear_frequency_signal import SignalLinearFrequency


class CompressedLFM:
    def para_init(self, origin_signal, compression_factor):
        self.origin_signal = copy.copy(origin_signal)
        self.compression_factor = compression_factor
        self.interfered_signal = numpy.zeros_like(self.origin_signal.time_domain)
        rect_function = numpy.where((-1e-5 <= self.origin_signal.t_range) & (self.origin_signal.t_range <= 1e-5), 1, 0)

        # Compress the pulse width
        compressed_t_duration = self.origin_signal.t_duration / compression_factor
        compressed_t_0 = self.origin_signal.t_0 + (1 - 1 / compression_factor) * (self.origin_signal.t_duration / 2)
        f_0 = self.origin_signal.f_0
        band = self.origin_signal.band
        t_duration = compressed_t_duration
        f_s = self.origin_signal.f_s
        t_0 = compressed_t_0
        # Generate M compressed and repeated LFM signals
        for i in range(compression_factor):
            compressed_signal = SignalLinearFrequency().para_init(
                f_0=self.origin_signal.f_0,
                band=self.origin_signal.band,
                t_duration=compressed_t_duration,
                f_s=self.origin_signal.f_s,
                t_0=compressed_t_0
            )
            compressed_signal_time_domain = compressed_signal.time_domain

            # Repeat the compressed signal at appropriate positions
            start_index = int(i * compressed_t_duration * self.origin_signal.f_s)
            end_index = start_index + len(compressed_signal_time_domain)
            self.interfered_signal[start_index:end_index] += compressed_signal_time_domain
        # self.interfered_signal =self.interfered_signal *rect_function
        return self


if __name__ == '__main__':
    num_signals = 10
    new_data_points = 2000  # 新的数据点数
    combined_array = numpy.empty((0, 2001), dtype=complex)
    for _ in range(num_signals):
        origin_signal = SignalLinearFrequency().para_init(f_0=50e6, band=15e6, t_duration=5e-5, f_s=40e6, t_0=-2.5e-5)
        signal = CompressedLFM().para_init(origin_signal=origin_signal,compression_factor=random.randint(3,10))
        JNR = random.randint(10, 100)
        max_amplitude = numpy.max(numpy.abs(signal.interfered_signal))
        signal.interfered_signal = signal.interfered_signal + 1 / (
                    10 ** (JNR / 10)) * max_amplitude * numpy.random.normal(0, 1, len(signal.origin_signal.t_range))
        label = 16
        modified_array = signal.interfered_signal.copy()
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
    plt.plot(signal.origin_signal.t_range, signal.interfered_signal.real)
    # plt.savefig('sstdomain.jpg', format='jpg')
    # plt.savefig('sstdomain.eps', format='eps', dpi=300)

    signal.f_domain = fft.fftshift(fft.fft(signal.interfered_signal))
    plt.figure(2)
    plt.plot(signal.origin_signal.f_range, numpy.abs(signal.f_domain))
    # plt.savefig('ssfdomain.jpg', format='jpg')
    # plt.savefig('ssfdomain.eps', format='eps', dpi=300)
    power_spectrum = numpy.abs(signal.f_domain) ** 2
    plt.figure(3)
    plt.specgram(signal.interfered_signal, Fs=origin_signal.f_s)
    plt.title('Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Intensity [dB]')
    # plt.savefig('sstfdomain.jpg', format='jpg')
    # plt.savefig('sstfdomain.eps', format='eps', dpi=300)

    plt.show()
