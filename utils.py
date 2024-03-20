import numpy
from base_signal import BaseSignal


def fill_list(list_a, filled_value=-100):
    narry = numpy.zeros([len(list_a), len(max(list_a, key=lambda x: len(x)))]) + filled_value
    for i, j in enumerate(list_a):
        narry[i][0:len(j)] = j
    return narry


def signal_conv(h: BaseSignal, x: BaseSignal):
    t_interval = x.t_range[1] - x.t_range[0]
    new_t_range = (numpy.array(range(h.time_domain.shape[0] - 1)) + 1) * t_interval + x.t_range[-1]
    t_range = numpy.concatenate((x.t_range, new_t_range), axis=0)
    time_domain = numpy.convolve(x.time_domain.real, h.time_domain.real)
    return BaseSignal().seq_init(t_range, time_domain)


def normalizing(input):
    min = numpy.min(input)
    result = (input-min)/numpy.max(input-min)
    return result
