import numpy
import numpy.fft as fft
from matplotlib import pyplot as plt
from base_signal import BaseSignal
from Radio_Noise import JammingDirectNoise
from scipy import interpolate
import pandas as pd
import random
class JammingNoiseFrequency(BaseSignal):
    def para_init(self, f_0, band, t_duration, f_s, t_0, with_window=False, **kwargs):
        self.carrier_amolitude = kwargs['carrier_amolitude']
        super().para_init(f_0, band, t_duration, f_s, t_0, with_window=False)
        return self

    def get_signal(self):
        output = JammingDirectNoise().para_init(f_0=self.f_0, band=self.band, t_duration=self.t_duration,
                                                f_s=self.f_s, t_0=-self.t_0).time_domain
        # output = self.band /10 * output
        kfm=random.uniform(0.5, 1.5)*self.band/10
        # w's integration
        frequency_phase = numpy.zeros(output.shape)
        for i in range(frequency_phase.shape[0] - 1):
            frequency_phase[i + 1] = frequency_phase[i] + (output[i] + output[i + 1]) / 2 * (self.t_range[1] - self.t_0)
        frequency_phase = 2 * numpy.pi *kfm* frequency_phase

        phase = 2 * numpy.pi * self.f_0 * self.t_range + frequency_phase
        return self.carrier_amolitude * numpy.exp(1j*phase)


if __name__ == '__main__':
    num_signals = 10
    new_data_points = 2000  # 新的数据点数
    combined_array = numpy.empty((0, 2001), dtype=complex)
    for _ in range(num_signals):
        signal = JammingNoiseFrequency().para_init(f_0=50e6, band=15e6, t_duration=5e-5, f_s=40e6, t_0=-2.5e-5, carrier_amolitude=random.random())
        JNR = random.randint(10, 100)
        max_amplitude = numpy.max(numpy.abs(signal.time_domain))
        signal.time_domain = signal.time_domain + 1 / (10 ** (JNR / 10)) * max_amplitude * numpy.random.normal(
            0, 1, len(signal.t_range))
        label = 3
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
    # plt.savefig('nfmtdomain.jpg', format='jpg')
    # plt.savefig('nfmtdomain.eps', format='eps', dpi=300)

    signal.f_domain = fft.fftshift(fft.fft(signal.time_domain))
    plt.figure(2)
    plt.plot(signal.f_range, numpy.abs(signal.f_domain))
    # plt.savefig('nfmfdomain.jpg', format='jpg')
    # plt.savefig('nfmfdomain.eps', format='eps', dpi=300)

    power_spectrum = numpy.abs(signal.f_domain) ** 2
    plt.figure(3)
    plt.specgram(signal.time_domain, Fs=signal.f_s)
    plt.title('Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Intensity [dB]')
    # plt.savefig('nfmtfdomain.jpg', format='jpg')
    # plt.savefig('nfmtfdomain.eps', format='eps', dpi=300)

    plt.show()
