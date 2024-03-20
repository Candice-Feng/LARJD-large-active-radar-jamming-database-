import numpy
import numpy.fft as fft
from matplotlib import pyplot as plt
from base_signal import BaseSignal
from filter import get_pass_filter
from scipy import interpolate
import pandas as pd
import random
import pickle

class JammingDirectNoise(BaseSignal):
    def get_signal(self):
        # 设定期望和方差
        mean_value = 0.0  # 期望（均值）
        variance_value = 1  # 方差
        noise = (numpy.random.normal(loc=mean_value, scale=numpy.sqrt(variance_value), size=self.t_range.shape) +
                 1j * numpy.random.normal(loc=mean_value, scale=numpy.sqrt(variance_value), size=self.t_range.shape))
        # rect_function = numpy.where((-1.5e-5 <= self.t_range) & (self.t_range <= 1.5e-5), 1, 0)

        # 将原始信号乘以矩形函数
        noise = noise

        noise_f = fft.fftshift(fft.fft(noise))
        filter = get_pass_filter(self.t_range.shape[0], self.f_s,self.t_0-self.band/2, self.t_0+self.band/2 )
        output = 2*noise_f * filter
        return fft.ifft(fft.ifftshift(output))


if __name__ == '__main__':
    num_signals = 10
    new_data_points = 2000  # 新的数据点数
    data_list = []
    combined_array = numpy.empty((0, 2001), dtype=complex)
    for _ in range(num_signals):
        signal = JammingDirectNoise().para_init(f_0=50e6, band=15e6*2, t_duration=5e-5, f_s=40e6, t_0=-2.5e-5)
        JNR = random.randint(10, 100)
        max_amplitude = numpy.max(numpy.abs(signal.time_domain))
        signal.time_domain = signal.time_domain + 1 / (10 ** (JNR / 10)) * max_amplitude * numpy.random.normal(
            0, 1, len(signal.t_range))

        normalized_jammer_signal = (signal.time_domain - numpy.min(signal.time_domain)) / (
                numpy.max(signal.time_domain) - numpy.min(signal.time_domain))
        # 提取实部
        real_part = numpy.real(normalized_jammer_signal)

        # 提取虚部
        imag_part = numpy.imag(normalized_jammer_signal)

        # # 提取幅度
        # magnitude = numpy.abs(normalized_jammer_signal)
        #
        # # 提取相位
        # phase = numpy.angle(normalized_jammer_signal)
        numpy.set_printoptions(threshold=numpy.inf)
        # 通过堆叠这些序列创建一个4xN矩阵
        blocker_matrix = numpy.vstack((real_part, imag_part))

        # 创建一个标签（对于干扰信号，标签为0）
        label = 1
        modified_array = signal.time_domain.copy()
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
    # plt.savefig('rntdomain.jpg', format='jpg')
    # plt.savefig('rntdomain.eps', format='eps', dpi=300)

    signal.f_domain = fft.fftshift(fft.fft(signal.time_domain))
    plt.figure(2)
    plt.plot(signal.f_range, numpy.abs(signal.f_domain))
    # plt.savefig('rnfdomain.jpg', format='jpg')
    # plt.savefig('rnfdomain.eps', format='eps', dpi=300)

    power_spectrum = numpy.abs(signal.f_domain) ** 2
    plt.figure(3)
    plt.specgram(signal.time_domain, Fs=signal.f_s)
    plt.title('Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Intensity [dB]')
    # plt.savefig('rntfdomain.jpg', format='jpg')
    # plt.savefig('rntfdomain.eps', format='eps', dpi=300)

    plt.show()
    # data = pd.read_csv('combined_data.csv')
    # df = pd.DataFrame()
    #
    # data_rows = []
    #
    # # 遍历列表中的每个ndarray元素，将ndarray展开为单个数，然后添加到矩阵中
    # for arr in generated_signals:
    #     flattened = arr.flatten()  # 展开ndarray
    #
    #     flattened_with_label = numpy.append(flattened, 1)  # 添加标签1
    #
    #     # 将flattened_with_label添加到data_rows列表
    #     data_rows.append(flattened_with_label)
    #
    # # 合并所有行
    # final_data = numpy.vstack(data_rows)
    #
    # # 将合并后的矩阵添加到DataFrame
    # df = pd.DataFrame(final_data, columns=[f'Column_{i + 1}' for i in range(final_data.shape[1])])
    #
    # df_real = df.applymap(lambda x: complex(x).real)
    #
    # # # Create DataFrame for the generated signals
    # # generated_data = pd.DataFrame({
    # #     'Signal': generated_signals,
    # #     'Label': 0
    # # })
    #
    # # Concatenate ISRJ and generated data
    # combined_data = pd.concat([data, df_real], ignore_index=True)
    # # 随机打乱数据
    # combined_data = combined_data.sample(frac=1).reset_index(drop=True)
    #
    # # 保存合并后的数据
    # file_name = 'combined_data.csv'
    # combined_data.to_csv(file_name, index=False)
    # print(f'Data saved to {file_name}')

