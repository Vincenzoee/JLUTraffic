import pickle
import sys
from itertools import zip_longest

import numpy as np
from matplotlib import pyplot as plt
from ruptures.base import BaseCost
import ruptures as rpt
#
# # 定义标志位，这里选择最高位作为邻接路段的标志位
# ADJACENCY_FLAG = 1 << 30


def number_to_decimal(number):
    # Separate the digits by dividing and taking modulo
    thousands = number // 1000
    hundreds = (number % 1000) // 100
    tens = (number % 100) // 10
    ones = number % 10

    # Construct the decimal number
    decimal_number = ones * (10 ** -1) + tens * (10 ** -2) + hundreds * (10 ** -3) + thousands * (10 ** -4)

    # Format the number to have exactly 4 decimal places, padding with zeros if necessary
    decimal_number = "{:.4f}".format(decimal_number)
    return float(decimal_number)


def reverse_decimal_to_integer(decimal):
    if decimal == int(decimal):
        return 0
    else:
        # 获取小数部分
        fractional_part = str(decimal).split('.')[1]
        # 逆序小数部分
        reversed_str = fractional_part[::-1]
        # Convert the reversed string back to an integer
        return int(decimal), int(reversed_str.lstrip('0'))

#
# def decode_road_ids(encoded_id: int) -> (bool, int, int):
#     # 检查邻接标志位
#     is_adjacent = bool(encoded_id & ADJACENCY_FLAG)
#
#     # 清除标志位
#     encoded_id &= ~ADJACENCY_FLAG
#
#     # 解码32位整数为两个16位整数
#     road_id1 = (encoded_id >> 16) & 0xffff
#     road_id2 = encoded_id & 0xffff
#
#     return is_adjacent, road_id1, road_id2


def split_connection_subseq(df, pair_list):
    # df = pickle.load(open(datapath, 'rb'))
    print(f'需转换{len(pair_list)}条路')
    df = df[['speed', 'acceleration','opath_list','cpath_list','interval','start_time']]
    # 假设df是你的DataFrame
    # 直接将字符串转换为浮点数，'nan'字符串会被转换为numpy.nan
    df['speed'] = df['speed'].apply(lambda x: np.array([float(item) for item in x]))
    df['interval'] = df['interval'].apply(lambda x: np.array([float(item) for item in x]))
    df['acceleration'] = df['acceleration'].apply(lambda x: np.array([float(item) for item in x]))

    # 将numpy.nan替换为0.0
    df['speed'] = df['speed'].apply(lambda x: np.nan_to_num(x, nan=0.0))
    df['interval'] = df['interval'].apply(lambda x: np.nan_to_num(x, nan=0.1))
    df['acceleration'] = df['acceleration'].apply(lambda x: np.nan_to_num(x, nan=0.0))

    # 定义自定义成本函数
    class CustomCost(BaseCost):
        model = "custom"
        min_size = 2

        def fit(self, signal):
            self.signal = signal
            return self

        def error(self, start, end):
            segment = self.signal[start:end]
            # 对加速度的从上过零点计算成本
            zero_crossings = ((segment[:-1, 1] > 0) & (segment[1:, 1] < 0)).sum()
            # 对速度的极值计算成本
            speed_diff = np.diff(segment[:, 0])
            local_extrema = np.abs(speed_diff[:-1] * speed_diff[1:]) <= 0
            return np.sum(zero_crossings) + np.sum(local_extrema)

    def get_change_points(sub_road_speed, sub_road_acceleration):
        # 将当前行的两个列表转换为一维数组，然后堆叠为二维数组
        speed_array = np.array(sub_road_speed, dtype=float)
        acceleration_array = np.array(sub_road_acceleration, dtype=float)

        # 使用numpy.column_stack按第一个维度堆叠数组，形成(2, current_length)形状的数组
        current_signal_array = np.column_stack((speed_array, acceleration_array))
        # 分别对速度和加速度数据进行变点检测

        series = current_signal_array
        # 创建并拟合自定义成本函数
        my_cost = CustomCost().fit(series)
        # model = 'l2'

        # 使用PELT算法进行变点检测
        algo = rpt.BottomUp(custom_cost=my_cost, min_size=5, jump=1).fit(series)
        # 检测变点，pen参数用于控制变点的惩罚，值越大惩罚越重,变点越少
        my_bkps = algo.predict(n_bkps=1)
        # fig, ax_array = rpt.display(series, my_bkps)
        # # 使用 suptitle 设置顶部标题
        # plt.suptitle(f"Change points: {my_bkps[0]}",  y=1.01)
        # print("Change points:", my_bkps)
        # plt.show()
        return my_bkps

    # def encode_road_ids(road_id1: int, road_id2: int) -> int:
    #     # 确保输入是16位整数
    #     assert 0 <= road_id1 < 2 ** 16
    #     assert 0 <= road_id2 < 2 ** 16
    #
    #     # 编码两个16位整数为一个32位整数，并设置邻接标志位
    #     encoded_id = ((road_id1 << 16) | road_id2) | ADJACENCY_FLAG
    #     return encoded_id

    df_length_list = []
    # 初始化road_interval和road_timestamp列，每个元素都是空列表
    df['road_interval'] = [[] for _ in df.index]
    df['road_timestamp'] = [[] for _ in df.index]
    for index, row in df.iterrows():
        length_list = []
        m = 0
        road_interval = []
        sub_road_start_time = row['start_time']
        road_start_time = [sub_road_start_time]
        for i in range(len(row['opath_list']) - 1):
            current_road = row['opath_list'][i]
            if row['opath_list'][i] != row['opath_list'][i + 1]:
                # When a different path is found, slice the DataFrame from m to i
                sub_road_speed = row['speed'][m:i + 1]
                sub_road_acceleration = row['acceleration'][m:i + 1]
                if (row['opath_list'][i], row['opath_list'][i + 1]) in set(pair_list) and len(sub_road_speed) > 9:
                    my_bkps = get_change_points(sub_road_speed, sub_road_acceleration)
                    # 在cpath_list中查找与当前路段相同的值
                    for j in range(len(row['cpath_list']) - 2, -1, -1):  # 逆向查找
                        if row['cpath_list'][j] == current_road and row['cpath_list'][j+1] == row['opath_list'][i + 1]:
                            # 计算新的小数部分
                            # 检查索引是否越界
                            if j + 1 >= len(row['cpath_list']):
                                print("索引越界错误: 尝试访问的索引超出了列表的范围")
                                print(j, current_road)
                                sys.exit()

                            # 创建新值
                            dec = number_to_decimal(row['cpath_list'][j + 1])
                            new_road = current_road+dec
                            # 插入新值到cpath_list中，紧挨着原始值之后
                            row['cpath_list'].insert(j + 1, new_road)
                            break
                    length_list.append(my_bkps[0])
                    length_list.append(my_bkps[1]-my_bkps[0])
                    start_index = i+1 - (my_bkps[1] - my_bkps[0])
                    # 替换值为new_road
                    row['opath_list'][start_index:i+1] = [new_road] * (i+1 - start_index)
                    sub_road_interval1 = sum(row['interval'][m:start_index])
                    sub_road_interval2 = sum(row['interval'][start_index:i+1])
                    sub_road_start_time += sub_road_interval1
                    road_interval.append(sub_road_interval1)
                    road_start_time.append(sub_road_start_time)
                    sub_road_start_time += sub_road_interval2
                    road_interval.append(sub_road_interval2)
                    road_start_time.append(sub_road_start_time)
                else:
                    length_list.append(len(sub_road_speed))
                    sub_road_interval = sum(row['interval'][m:i + 1])
                    sub_road_start_time += sub_road_interval
                    road_interval.append(sub_road_interval)
                    road_start_time.append(sub_road_start_time)
                m = i + 1
        # Don't forget to add the last sub_road after the loop
        length_list.append(len(row['speed'][m:]))
        sub_road_interval = sum(row['interval'][m:])
        sub_road_start_time += sub_road_interval
        road_interval.append(sub_road_interval)
        road_start_time.append(sub_road_start_time)
        # # add_road_connection_field(row)
        # print(row['cpath_list'])  # [5155, 5155.5, 5156, 5159, 5126, 5126.46, 5122]
        # print(row['opath_list'])
        # print(row['road_connection'])  # [(5155, 0), (5155, 5156), (5156, 0), (5159, 0), (5126, 0), (5126, 5122), (5122, 0)]
        # print(length_list)  # [7, 7, 1, 6, 6, 7, 15, 5]
        df_length_list.append(length_list)
        # print('df_length_list', df_length_list)
        df['road_interval'][index] = road_interval
        df['road_timestamp'][index] = road_start_time
        # print(row['start_time'],row['interval'], row['road_interval'],row['road_timestamp'])
    # # 确定内部列表的最大长度
    # max_length = max(len(item) for item in df_length_list)
    # # 使用列表推导式填充每个内部列表
    # padded_gps_list = [
    #     sublist + [0] * (max_length - len(sublist)) for sublist in df_length_list
    # ]
    # print(padded_gps_list)
    return df_length_list, df


# datapath = r"data/JGRM_DATASET/chengdu/chengdu_1101_1115_500.pkl"
# length=split_connection_subseq(datapath)



