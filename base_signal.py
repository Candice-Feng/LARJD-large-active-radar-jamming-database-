import abc
import numpy


class BaseSignal:
    def para_init(self, f_0, band, t_duration, f_s, t_0, with_window=False):
        assert f_0 >= band / 2
        self.f_0 = f_0 #中心频率
        self.band = band #带宽
        self.t_duration = t_duration #持续时间
        self.f_s = f_s #采样频率
        self.t_0 = t_0 #起始时间
        self.with_window = with_window
        self.timestep = 1 / f_s #采样周期
        self.t_range, self.f_range = self.get_range()
        self.time_domain = self.get_signal()
        return self

    def seq_init(self, t_range, time_domain, with_window=False):
        self.t_range = t_range
        self.time_domain = time_domain
        self.with_window = with_window

        self.t_duration = t_range[-1] - t_range[0]
        num_point = self.t_range.shape[0]
        self.f_s = num_point / self.t_duration

        self.f_range = numpy.linspace(-self.f_s / 2, self.f_s / 2 - self.f_s / num_point, num_point)
        return self

    def get_range(self):
        # 此处通过缩小f_s可以消除四舍五入误差。同时频域的值域不变，但是，定义域会缩短。因此，round实际上会引入频率方面的误差。
        # 当通过下方公式倒推f_s时，会导致f_s有偏差，不过这不影响值域和点数。
        num_point = round(self.t_duration * self.f_s)
        t_range = numpy.linspace(self.t_0, self.t_0 + self.t_duration, num_point)
        f_range = numpy.linspace(-self.f_s / 2, self.f_s / 2 - self.f_s / num_point, num_point)
        return t_range, f_range

    @abc.abstractmethod
    def get_signal(self):
        pass

    def get_match_filter(self):
        if self.with_window:
            match_filter = numpy.fliplr(
                (self.time_domain.real * numpy.hanning(self.time_domain.shape[0]))[numpy.newaxis, :])
        else:
            match_filter = numpy.fliplr(self.time_domain.real[numpy.newaxis, :])
        return match_filter[0]

    def match_filtering(self, x):
        t_interval = x.t_range[1] - x.t_range[0]
        new_t_range = (numpy.array(range(self.get_match_filter().shape[0] - 1)) + 1) * t_interval + x.t_range[-1]
        t_range = numpy.concatenate((x.t_range, new_t_range), axis=0)
        time_domain = numpy.convolve(x.time_domain.real, self.get_match_filter())
        return BaseSignal().seq_init(t_range, time_domain)
