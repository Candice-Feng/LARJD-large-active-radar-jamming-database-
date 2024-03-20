import pandas as pd
import matplotlib.pyplot as plt
import numpy

# 读取.tsv文件
file_path = './train_data_list.tsv'  # 替换为你的文件路径
column_names = [str(i) for i in range(2001)]
df = pd.read_csv(file_path, sep='\t', names=column_names)
for column in df.columns:
    if df[column].dtype == numpy.dtype('O'):
        # 如果列的数据类型是 'O' (Object)，则进行转换
        df[column] = df[column].apply(lambda x: complex(x))
# 设置横坐标范围和间隔
x_range = numpy.linspace(-2.5e5, 2.5e5, 2000)
# 获取第一行第二列的矩阵
blocker_matrix_data = df.iloc[0, 1:]
real_part = blocker_matrix_data.apply(numpy.real)
imag_part = blocker_matrix_data.apply(numpy.imag)


# 绘制图形
plt.plot(x_range, real_part)
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Plot of the First Row of Matrix Data')
plt.show()
