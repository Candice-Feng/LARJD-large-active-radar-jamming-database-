import copy
import numpy
import numpy.fft as fft
from matplotlib import pyplot as plt
from lfm import SignalLinearFrequency

class Volicitydelay:
    def para_init(self, origin_signal):
        self.origin_signal = copy.copy(origin_signal)
        f_s = self.origin_signal.f_s
        band = self.origin_signal.band
        t_duration = self.origin_signal.t_duration
        t_0 = self.origin_signal.t_0
        f_domain = fft.fftshift(fft.fft(self.origin_signal.time_domain))
        new_f_domian=1.5*f_domain
        delay_time = t_duration/5
        self.interfered_signal = numpy.zeros_like(f_domain)
        delay_index = int(delay_time * f_s)
        sample_index = int(t_duration/5 * f_s)
        start = int(self.origin_signal.t_0 * f_s)
        # Identify indices corresponding to the time window
        start_index = start
        end_index = start_index+int(t_duration * f_s)-int(t_duration/5 * f_s)
        start_direct_index = start_index+delay_index
        end_direct_index = end_index + delay_index

        # Perform sampling
        sampled_signal = new_f_domian[start_index:end_index]

        # Directly forward the sampled signal
        self.interfered_signal[start_direct_index:end_direct_index] = sampled_signal
        self.interfered_signal = fft.ifft(fft.ifftshift( self.interfered_signal))

        return self


if __name__ == '__main__':

    origin_signal = SignalLinearFrequency().para_init(f_0=5e6 / 2, band=5e6, t_duration=5e-5, f_s=40e6, t_0=-0)
    signal = Volicitydelay().para_init(origin_signal=origin_signal)

    plt.figure(1)
    plt.plot(signal.origin_signal.t_range, signal.interfered_signal.real)

    signal.f_domain = fft.fftshift(fft.fft(signal.interfered_signal))
    plt.figure(2)
    plt.plot(signal.origin_signal.f_range, numpy.abs(signal.f_domain))
    power_spectrum = numpy.abs(signal.f_domain) ** 2
    plt.figure(3)
    plt.specgram(signal.interfered_signal, Fs=origin_signal.f_s)

    plt.show()

