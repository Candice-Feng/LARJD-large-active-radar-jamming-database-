import numpy


def get_pass_filter(num_point, f_s, f_start, f_end):
    filter = numpy.zeros(num_point)
    for i in range(round((f_start+f_s/2)/(f_s/num_point))+1, round((f_end+f_s/2)/(f_s/num_point))+1):
        filter[i] = 1
    return filter
