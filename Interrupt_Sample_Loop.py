import copy
import numpy
import numpy.fft as fft
from matplotlib import pyplot as plt
from linear_frequency_signal import SignalLinearFrequency
import random
import pandas as pd
class Interruptsampleloop:
    def para_init(self, origin_signal):
        self.origin_signal = copy.copy(origin_signal)
        self.interfered_signal = numpy.zeros_like(self.origin_signal.time_domain)
        f_s = self.origin_signal.f_s
        band = self.origin_signal.band
        t_duration = self.origin_signal.t_duration
        t_0 = self.origin_signal.t_0
        samplenum = random.randint(6,7)  # 要累加的自然数的最大值
        slicenum = 0
        startslice = 0
        for i in range(1,samplenum+1):
            slicenum += i+1
        slice_duration = t_duration / slicenum
        slicecut = int(slice_duration * f_s)
        sampled_signal = []

        for i in range(1,samplenum+1):
            startslice += i+1
            sum1 = sum(range(i))
            samplelocation = i-1+sum1
            extract = i+sum1
            sum2 = i+2

            start_index = int(t_0 * f_s + samplelocation * slicecut)
            end_index = int(start_index + slicecut)
            # sampled_signal[i] = self.origin_signal.time_domain[start_index:end_index]
            sampled_signal.append(list(self.origin_signal.time_domain[start_index:end_index]))

            # extract = slice
            # extract_start_time = end_index
            # extract_end_time = end_index + (startslice+1) * slicecut

        for i in range(1, samplenum + 1):
            startslice += i + 1
            extract1 = 0 + sum(range(i + 1))
            cut = extract1 * slicecut
            if cut < 2000:
               self.interfered_signal[int(t_0 * f_s + extract1 * slicecut)-1000:int(t_0 * f_s + extract1 * slicecut)+slicecut-1000] += sampled_signal[0]
            else:
                pass
        for i in range(1, samplenum + 1):
            extract2 = 1 + sum(range(i + 2))
            cut = extract2* slicecut
            if cut < 2000:
               self.interfered_signal[int(t_0 * f_s + extract2 * slicecut) -1000:int(t_0 * f_s + extract2 * slicecut) + slicecut -1000] += sampled_signal[1]
            else:
                pass
        for i in range(1, samplenum + 1):
            extract3 = 2 + sum(range(i + 3))
            cut = extract3* slicecut
            if cut < 1999:
               self.interfered_signal[int(t_0 * f_s + extract3 * slicecut) -1000:int(t_0 * f_s + extract3 * slicecut) + slicecut-1000] += sampled_signal[2]
            else:
                pass
        for i in range(1, samplenum + 1):
            extract4 = 3 + sum(range(i + 4))
            cut = extract4 * slicecut
            if cut < 1999:
               self.interfered_signal[int(t_0 * f_s + extract4 * slicecut) -1000:int(t_0 * f_s + extract4 * slicecut) + slicecut -1000] += sampled_signal[3]
            else:
                pass

        for i in range(1, samplenum + 1):
            extract5 = 4 + sum(range(i + 5))
            cut = extract5 * slicecut
            if cut < 1999:
               self.interfered_signal[int(t_0 * f_s + extract5 * slicecut) -1000:int(t_0 * f_s + extract5 * slicecut) + slicecut -1000] += sampled_signal[4]
            else:
                pass

        for i in range(1, samplenum + 1):
            extract6 = 5 + sum(range(i + 6))
            cut = extract6 * slicecut
            if cut < 2000:
               self.interfered_signal[int(t_0 * f_s + extract6 * slicecut) -1000:int(t_0 * f_s + extract6 * slicecut) + slicecut-1000 ] += sampled_signal[5]
            else:
                pass

        return self


if __name__ == '__main__':
    num_signals = 10
    new_data_points = 2000  # 新的数据点数
    combined_array = numpy.empty((0, 2001), dtype=complex)
    for _ in range(num_signals):

      origin_signal = SignalLinearFrequency().para_init(f_0=50e6, band=15e6, t_duration=5e-5, f_s=40e6, t_0=-2.5e-5)
      signal = Interruptsampleloop().para_init(origin_signal=origin_signal)
      JNR = random.randint(10, 100)
      max_amplitude = numpy.max(numpy.abs(signal.interfered_signal))
      signal.interfered_signal = signal.interfered_signal + 1 / (10 ** (JNR / 10)) * max_amplitude * numpy.random.normal(0,1,len(signal.origin_signal.t_range))
      label = 15
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
    # plt.savefig('isltdomain.jpg', format='jpg')
    # plt.savefig('isltdomain.eps', format='eps', dpi=300)

    signal.f_domain = fft.fftshift(fft.fft(signal.interfered_signal))
    plt.figure(2)
    plt.plot(signal.origin_signal.f_range, numpy.abs(signal.f_domain))
    # plt.savefig('islfdomain.jpg', format='jpg')
    # plt.savefig('islfdomain.eps', format='eps', dpi=300)
    power_spectrum = numpy.abs(signal.f_domain) ** 2
    plt.figure(3)
    plt.specgram(signal.interfered_signal, Fs=origin_signal.f_s)
    plt.title('Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Intensity [dB]')
    # plt.savefig('isltfdomain.jpg', format='jpg')
    # plt.savefig('isltfdomain.eps', format='eps', dpi=300)

    plt.show()

