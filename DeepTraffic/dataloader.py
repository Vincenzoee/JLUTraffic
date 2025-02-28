#!/usr/bin/python
import ast
import math
import os
from collections import namedtuple
from decimal import getcontext, Decimal
from glob import glob

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset, Sampler
import torch.nn.utils.rnn as rnn_utils
import pickle
import pandas as pd
from split_subseq import split_connection_subseq,number_to_decimal
import torch.nn as nn
import numpy as np
import torch
import warnings
from datetime import datetime
from tqdm import tqdm
from graph_utils.data_processing import compute_time_statistics
from graph_utils.utils import get_neighbor_finder, RandEdgeSampler
warnings.filterwarnings('ignore')


def extract_weekday_and_minute_and_second_from_list(timestamp_list):
    weekday_list = []
    minute_list = []
    second_list = []
    for timestamp in timestamp_list:
        dt = datetime.fromtimestamp(timestamp)
        weekday = dt.weekday()  # 0-6, Monday is 0
        minute = dt.hour * 60 + dt.minute
        second = dt.second
        weekday_list.append(weekday / 6.0)  # Normalize to 0-1 range, where Sunday is 0 and Saturday is 1
        minute_list.append(minute / 1440.0)  # Normalize minutes to 0-1 range
        second_list.append(second / 59.0)  # Normalize seconds to 0-1 range within a minute, note that 59 is the max second value
    weekday_tensor = torch.tensor(weekday_list).float()
    minute_tensor = torch.tensor(minute_list).float()
    second_tensor = torch.tensor(second_list).float()
    return np.vstack([weekday_tensor.numpy(), minute_tensor.numpy(), second_tensor.numpy()]).T


def prepare_gps_data(df, mat_padding_value, data_padding_value, max_len):
    """
    Args:
        df: cpath_list,
        mat_padding_value: default num_nodes
        data_padding_value: default 0
    Returns:
        gps_data: (batch, gps_max_length, num_features)
        gps_assign_mat: (batch, gps_max_length)
    """
    # 直接将字符串转换为浮点数，'nan'字符串会被转换为numpy.nan
    df['angle_delta']= df['angle_delta'].apply(lambda x: np.array([float(item) for item in x]))
    df['dist'] = df['dist'].apply(lambda x: np.array([float(item) for item in x]))

    # 将numpy.nan替换为0.0
    df['angle_delta'] = df['angle_delta'].apply(lambda x: np.nan_to_num(x, nan=0.0))
    df['dist'] = df['dist'].apply(lambda x: np.nan_to_num(x, nan=0.0))
    # 提取特征
    tm_list = df['tm_list'].tolist()  # 单独处理
    # lng_list = df['lng_list'].tolist()  # 1
    # lat_list = df['lat_list'].tolist()  # 1
    speed = df['speed'].tolist()  # 内生
    acceleration = df['acceleration'].tolist()  # 1
    angle_delta = df['angle_delta'].tolist()  # 1
    interval = df['interval'].tolist()
    dist = df['dist'].tolist()
    opath_list = df['opath_list'].tolist()
    # 确保三个大列表的长度相同
    if len(angle_delta) == len(dist) == len(interval):
        # 使用嵌套的列表推导式对每个小列表中的对应元素相除
        per_angle = [[a / i if i != 0 else a/0.1 for a, i in zip(sub_angle_delta, sub_interval)]
                     for sub_angle_delta, sub_interval in zip(angle_delta, interval)]
        per_dist = [[d / i if i != 0 else d/0.1 for d, i in zip(sub_dist, sub_interval)]
                    for sub_dist, sub_interval in zip(dist, interval)]
    else:
        print("大列表的长度不一致，无法进行元素对应相除。")
        return None

    # 组合成特征矩阵
    endo_features = []
    exo_features = []
    for i in range(len(tm_list)):
        feature_row = [
            torch.tensor(speed[i], dtype=torch.float64),
        ]
        endo_features.append(torch.stack(feature_row, dim=1))
    for i in range(len(tm_list)):
        feature_row = [
            torch.tensor(acceleration[i], dtype=torch.float64),
            torch.tensor(per_angle[i], dtype=torch.float64),
            # torch.tensor(per_dist[i], dtype=torch.float64),
            # torch.tensor(lng_list[i], dtype=torch.float64),
            # torch.tensor(lat_list[i], dtype=torch.float64)
        ]
        exo_features.append(torch.stack(feature_row, dim=1))

    # 路段ID转换为张量
    opath_list_tensors = [torch.tensor(opath_list, dtype=torch.float64) for opath_list in opath_list]

    # 划分输入和目标序列
    input_gps_time_feature = []
    target_gps_time_feature = []
    input_endo_features = []
    input_exo_features = []
    target_endo_features = []
    input_opath = []
    target_opath = []
    # 提取时间特征
    # time_features_list = [extract_weekday_and_minute_and_second_from_list(timestamps) for timestamps in tm_list]

    for gps_time_feature, endo_feature, exo_feature, opath in zip(tm_list, endo_features,
                                                                  exo_features, opath_list_tensors):
        split_index = int(len(endo_feature) * 0.7)  # 70%作为输入序列

        input_gps_time_feature.append(gps_time_feature[:split_index])
        input_endo_features.append(endo_feature[:split_index])
        input_exo_features.append(exo_feature[:split_index])
        input_opath.append(opath[:split_index])

        target_endo_features.append(endo_feature[split_index:])
        target_opath.append(opath[split_index:])
        target_gps_time_feature.append(gps_time_feature[split_index:])
    # Padding 输入序列
    # 使用 pad_sequence 进行填充
    x_indo_features_padded = rnn_utils.pad_sequence(input_endo_features, padding_value=0.0, batch_first=True)
    x_exo_features_padded = rnn_utils.pad_sequence(input_exo_features, padding_value=0.0, batch_first=True)
    x_opath_padded = rnn_utils.pad_sequence(input_opath, padding_value=-1, batch_first=True)
    input_gps_time_feature = [torch.from_numpy(np.array(x)) for x in input_gps_time_feature]
    x_time_feature_padded = rnn_utils.pad_sequence(input_gps_time_feature, padding_value=-2,
                                                   batch_first=True)

    # Padding 目标序列
    y_indo_features_padded = rnn_utils.pad_sequence(target_endo_features, padding_value=0.0,
                                                    batch_first=True)
    y_opath_padded = rnn_utils.pad_sequence(target_opath, padding_value=-1, batch_first=True)
    target_gps_time_feature = [torch.from_numpy(np.array(x)) for x in target_gps_time_feature]
    y_time_feature_padded = rnn_utils.pad_sequence(target_gps_time_feature, padding_value=-2,
                                                   batch_first=True)

    return x_time_feature_padded, x_indo_features_padded, x_exo_features_padded, x_opath_padded, \
        y_time_feature_padded, y_indo_features_padded, y_opath_padded

    # # padding opath_list
    # opath_list = [torch.tensor(opath_list, dtype=torch.float64) for opath_list in df['opath_list']]
    #
    # gps_assign_mat = rnn_utils.pad_sequence(opath_list, padding_value=mat_padding_value, batch_first=True)
    # # padding gps point data
    # data_package = []
    # for col in df.drop(columns='opath_list').columns:
    #     features = df[col].tolist()
    #     features = [torch.tensor(f, dtype=torch.float32) for f in features]
    #     features = rnn_utils.pad_sequence(features, padding_value=torch.nan, batch_first=True)
    #     features = features.unsqueeze(dim=2)
    #     data_package.append(features)
    #
    # gps_data = torch.cat(data_package, dim=2)
    #
    # # todo 临时处理的方式, 把时间戳那维特征置1
    # gps_data[:, :, 0] = torch.ones_like(gps_data[:, :, 0])
    #
    # # 对除第一维特征进行标准化
    # for i in range(1, gps_data.shape[2]):
    #     fea = gps_data[:, :, i]
    #     nozero_fea = torch.masked_select(fea, torch.isnan(fea).logical_not())  # 计算不为nan的值的fea的mean与std
    #     gps_data[:, :, i] = (gps_data[:, :, i] - torch.mean(nozero_fea)) / torch.std(nozero_fea)
    #
    # # 把因为数据没有前置节点因此无法计算，加速度等特征的nan置0
    # gps_data = torch.where(torch.isnan(gps_data), torch.full_like(gps_data, data_padding_value), gps_data)
    #
    # return gps_data, gps_assign_mat


def prepare_route_data(df,mat_padding_value,data_padding_value,max_len):
    """

    Args:
        df: cpath_list,
        mat_padding_value: default num_nodes
        data_padding_value: default 0

    Returns:
        route_data: (batch, route_max_length, num_features)
        route_assign_mat: (batch, route_max_length)

    """
    # padding capath_list
    cpath_list = [torch.tensor(cpath_list, dtype=torch.float64) for cpath_list in df['cpath_list']]
    route_assign_mat = rnn_utils.pad_sequence(cpath_list, padding_value=mat_padding_value,
                                                               batch_first=True)

    # padding route data
    weekday_route_list, minute_route_list = zip(
        *df['road_timestamp'].apply(extract_weekday_and_minute_and_second_from_list))

    weekday_route_list = [torch.tensor(weekday[:-1]).long() for weekday in weekday_route_list]  # road_timestamp 比 route 本身的长度多1，包含结束的时间戳
    minute_route_list = [torch.tensor(minute[:-1]).long() for minute in minute_route_list]  # road_timestamp 比 route 本身的长度多1，包含结束的时间戳
    weekday_data = rnn_utils.pad_sequence(weekday_route_list, padding_value=0, batch_first=True)
    minute_data = rnn_utils.pad_sequence(minute_route_list, padding_value=0, batch_first=True)

    # 分箱编码时间值
    # interval = []
    # df['road_interval'].apply(lambda row: interval.extend(row))
    # interval = np.array(interval)[~np.isnan(interval)]
    #
    # cuts = np.percentile(interval, [0, 2.5, 16, 50, 84, 97.5, 100])
    # cuts[0] = -1
    #
    # new_road_interval = []
    # for interval_list in df['road_interval']:
    #     new_interval_list = pd.cut(interval_list, cuts, labels=[1, 2, 3, 4, 5, 6])
    #     new_road_interval.append(torch.Tensor(new_interval_list).long())

    new_road_interval = []
    for interval_list in df['road_interval']:
        new_road_interval.append(torch.Tensor(interval_list).long())

    delta_data = rnn_utils.pad_sequence(new_road_interval, padding_value=-1, batch_first=True)

    route_data = torch.cat([weekday_data.unsqueeze(dim=2), minute_data.unsqueeze(dim=2), delta_data.unsqueeze(dim=2)], dim=-1)  # (batch_size,max_len,3)

    # 填充nan
    route_data = torch.where(torch.isnan(route_data), torch.full_like(route_data, data_padding_value), route_data)

    return route_data, route_assign_mat


class NodeEmbeddingModel(nn.Module):
    def __init__(self, vocab_size=None, road_embed_size=None, init_road_emb=None, edge_features_df=None, embedding_layer=None):
        super(NodeEmbeddingModel, self).__init__()
        if embedding_layer is not None:
            self.node_embedding = embedding_layer
        else:
            assert vocab_size is not None and road_embed_size is not None, "Both vocab_size and embed_size must be provided when embedding_layer is None"
            self.node_embedding = torch.nn.Embedding(vocab_size, road_embed_size)
            self.vocab_size = vocab_size
            self.road_embed_size = road_embed_size
            self.start_id = vocab_size
            self.nodes_dict = {}
            # 加载预训练的节点嵌入权重
            self.node_embedding.weight = torch.nn.Parameter(init_road_emb['init_road_embd'])
            self.node_embedding.requires_grad_(True)
            # 加载路段属性
            self.edge_features_df = edge_features_df
            self.road_attr_size = len(edge_features_df.columns) - 1  # 减去 ID 列

            # 定义一个线性层来处理路段属性
            self.attr_linear = nn.Linear(self.road_attr_size, self.road_embed_size)
            # 线性投影层用于类型信息
            self.type_linear = nn.Linear(1, self.road_embed_size)

    def get_road_attributes(self, road_id):
        # 从 DataFrame 中获取指定 road_id 的属性
        row = self.edge_features_df[self.edge_features_df['fid'] == road_id]
        if not row.empty:
            attributes = row.iloc[0].drop('fid').values.astype(float)
            return torch.tensor(attributes).float()
        else:
            raise ValueError(f"Road ID {road_id} not found in edge_features_df.")

    def interpolate_features(self, from_node, to_node):
        """根据已有节点的特征向量和路段属性，估计新节点的特征向量"""
        from_node_feature = self.node_embedding.weight[from_node]
        to_node_feature = self.node_embedding.weight[to_node]

        # 获取路段属性
        from_node_attr = self.get_road_attributes(from_node)
        to_node_attr = self.get_road_attributes(to_node)

        # 处理路段属性
        from_node_attr_emb = self.attr_linear(from_node_attr).to('cuda')
        to_node_attr_emb = self.attr_linear(to_node_attr).to('cuda')

        # 结合节点嵌入和属性嵌入
        from_node_combined = from_node_feature + from_node_attr_emb
        to_node_combined = to_node_feature + to_node_attr_emb

        # 插值
        new_node_feature = (from_node_combined + to_node_combined) / 2
        return new_node_feature

    def add_new_nodes(self, edge_index, unique_nodes, device='cuda'):
        new_node_emb = []
        pair_nodes = []
        for from_node, to_node in zip(edge_index[0], edge_index[1]):
            # 检查 node 是否不在 unique_nodes 列表中
            if from_node.item() not in unique_nodes:
                # 在 edge_feature 中查找与 node 匹配的 fid，检查对应的 length 是否大于 100
                condition = self.edge_features_df[self.edge_features_df['fid'] == from_node.item()]['length_id'] > 0
                # 如果两个条件都满足，执行操作
                if condition.any():
                    dec = number_to_decimal(to_node)
                    new_road = round(from_node.item() + dec, 4)
                    self.nodes_dict[new_road] = self.start_id
                    self.start_id += 1
                    pair_nodes.append((from_node.item(), to_node.item()))
                    new_node_emb.append(self.interpolate_features(from_node.item(), to_node.item()))

        # 更新节点嵌入层
        self.update_embedding(new_node_emb, device)

        # 将字典保存到文件
        with open('./data/1w_nodes_dict.pkl', 'wb') as f:
            pickle.dump(self.nodes_dict, f)
        with open('./data/1w_pair_nodes.pkl', 'wb') as f:
            pickle.dump(pair_nodes, f)
        print('新旧节点对应关系保存成功')
        return pair_nodes

    def update_embedding(self, new_node_emb, device):
        # 更新旧节点的特征向量，使其也包含路段属性
        # 首先获取所有节点的属性
        all_node_attributes = [self.get_road_attributes(i) for i in range(self.vocab_size)]
        # 将所有节点属性转换为一个张量
        all_node_attributes = torch.stack(all_node_attributes, dim=0)
        # 将所有节点属性转换为嵌入向量
        all_attr_embs = self.attr_linear(all_node_attributes).to(device)
        old_nodes_type = torch.zeros(self.vocab_size, dtype=torch.float32).unsqueeze(dim=1)
        old_type_embs = self.type_linear(old_nodes_type).to(device)
        # 使用 no_grad 上下文管理器来避免不必要的梯度计算
        with torch.no_grad():
            # 克隆现有的权重
            weight_clone = self.node_embedding.weight.clone().to(device)
            # 执行加法操作
            weight_clone += all_attr_embs + old_type_embs
            # 将更新后的权重转换为 Parameter 并赋值回去
            self.node_embedding.weight = torch.nn.Parameter(weight_clone)

        num_new_nodes = self.start_id - self.vocab_size
        new_vocab_size = self.start_id
        new_nodes_type = torch.ones(num_new_nodes, dtype=torch.float32).unsqueeze(dim=1)
        new_type_embs = self.type_linear(new_nodes_type).to(device)
        # 创建新的节点嵌入层，其大小等于原大小加上新节点的数量
        new_node_embedding = nn.Embedding(new_vocab_size, self.road_embed_size).to(device)
        # 复制旧的节点嵌入到新的层中
        with torch.no_grad():
            new_node_embedding.weight[:new_vocab_size - num_new_nodes] = self.node_embedding.weight.clone()
        # 初始化新节点的嵌入权重
            # 将类型信息嵌入向量与新节点的原始嵌入向量相加
            new_node_emb_tensors_with_types = [emb + type_emb for emb, type_emb in
                                               zip(new_node_emb, new_type_embs)]
        new_node_emb_tensor = torch.stack(new_node_emb_tensors_with_types)
        with torch.no_grad():
            # 明确指出是从旧节点的末尾开始，直到新的vocab_size结束
            new_node_embedding.weight[-num_new_nodes:] = new_node_emb_tensor

        # 替换旧的节点嵌入层
        self.node_embedding = new_node_embedding
        self.node_embedding.requires_grad_(True)

    def save_embedding(self, filepath):
        """
        保存模型的state_dict到指定的文件路径。
        :param filepath: str, 模型保存的文件路径
        """
        torch.save(self.state_dict(), filepath)


def generate_temp_edge(df, prefix, edge_features_df):
    # 加载字典
    with open('./data//1w_nodes_dict.pkl', 'rb') as f:
        nodes_dict = pickle.load(f)
    # 创建一个空的DataFrame来存储结果
    result_df = pd.DataFrame(columns=['source', 'destination', 'label'])

    # 遍历原始DataFrame的每一条记录
    for index, row in df.iterrows():
        cpath_list = row['cpath_list']
        # flag = row['flag']

        # 按照描述创建路段变换序列
        for i in range(len(cpath_list) - 1):
            source = cpath_list[i]
            destination = cpath_list[i + 1]

            # 将数据添加到结果DataFrame中
            result_df = pd.concat([result_df, pd.DataFrame({'source': [source],
                                                            'destination': [destination],
                                                            'label': [index]})], ignore_index=True)
    edge_feature_gen = EdgeFeatureGenerator(edge_features_df)
    edge_feature_gen.process_edges(result_df, prefix)
    # 直接遍历每一行，更新source和destination'
    for index, row in result_df.iterrows():
        result_df.at[index, 'source'] = nodes_dict.get(row['source'], row['source'])
        result_df.at[index, 'destination'] = nodes_dict.get(row['destination'], row['destination'])
    result_df['source'] = result_df['source'].astype(int)
    result_df['destination'] = result_df['destination'].astype(int)
    result_df.to_csv(f'./data/{prefix}+temp_edge.csv', index=False)
    print(f"{prefix}+temp_edge.csv saved")

    # def update_edge_index(self, from_node, to_node, new_node):
    #     # 查找并验证要替换的边是否存在
    #     edge_to_replace_mask = (self.edge_index[0, :] == from_node) & (self.edge_index[1, :] == to_node)
    #     if not edge_to_replace_mask.any():
    #         print(f"No edge found from node {from_node} to node {to_node}.")
    #         return
    #
    #     # 确认只有一条边需要被替换
    #     assert edge_to_replace_mask.sum() == 1, "Multiple edges found from node {} to node {}. This function supports replacing only one edge at a time.".format(
    #         from_node, to_node)
    #     index_to_replace = torch.where(edge_to_replace_mask)[0][0]
    #
    #
    #     # 准备新边的数据
    #     new_edges = torch.tensor([[from_node, new_node],
    #                               [new_node, to_node]],
    #                              dtype=self.edge_index.dtype,
    #                              device=self.edge_index.device)
    #
    #     # 更新edge_index
    #     # 1. 截取需要保留的前半部分边
    #     # 2. 将新边插入到指定位置（替换原位置的边）
    #     # 3. 将剩余的边追加到新边之后
    #     updated_edges = torch.cat([self.edge_index[:, :index_to_replace],
    #                                new_edges,
    #                                self.edge_index[:, index_to_replace + 1:]],
    #                               dim=1)
    #
    #     self.edge_index = updated_edges


class EdgeFeatureGenerator:
    def __init__(self, edge_features_df=None):
        self.edge_type_to_feature = {}
        self.edge_features_df = edge_features_df

    # def edge_type(self, node1, node2):
    #     """根据两个节点的类型确定边的类型"""
    #     is_straight_node1 = isinstance(node1, int) or (isinstance(node1, float) and node1.is_integer())
    #     is_straight_node2 = isinstance(node2, int) or (isinstance(node2, float) and node2.is_integer())
    #
    #     type1 = 'straight' if is_straight_node1 else 'turn'
    #     type2 = 'straight' if is_straight_node2 else 'turn'
    #
    #     return f'{type1}-{type2}'

    # def generate_edge_feature(self, node1, node2):
        """为每种边类型生成128维的特征向量"""
        # 定义一个线性层来处理路段属性
        attr_linear = nn.Linear(6, 128)

        # # 归一化车道数之差
        # lane_diff_normalized = (node1['lanes'] - node2['lanes']+1) / 2
        # length_diff_normalized = (node1['length'] - node2['length']+1) / 2
        # feature_vector = [
        #     int(node1['oneway']),  # Oneway (always 1 in this case)
        #     lane_diff_normalized,  # Difference in lanes between source and target
        #     int(node1['bridge'] or node2['bridge']),  # Bridge (1 if either has bridge)
        #     int(node1['tunnel'] or node2['tunnel']),  # Tunnel (1 if either has tunnel)
        #     int(node1['highway_id'] == node2['highway_id']),
        #     # Whether highway types match (1 if same, 0 otherwise)
        #     length_diff_normalized,  # Length ID of source node (could be adjusted)
        # ]
        # feature_vector = torch.tensor(feature_vector, dtype=torch.float32)
        # feature_vector = attr_linear(feature_vector)
        # return feature_vector

    def process_edges(self, result_df, prefix):

        all_node_features = self.edge_features_df.set_index('fid')

        # 假设 result_df 包含 'source' 和 'destination' 列
        sources = result_df['source'].apply(int).values
        destinations = result_df['destination'].apply(int).values

        # 获取所有源节点和目标节点的特征
        source_features = all_node_features.loc[sources].values
        destination_features = all_node_features.loc[destinations].values

        # 初始化边特征矩阵
        edge_features_matrix = []

        # 从源节点特征中获取 oneway 属性值
        oneway_values = source_features[:, all_node_features.columns.get_loc('oneway')]

        # 计算车道数之差
        lane_diff = source_features[:, all_node_features.columns.get_loc('lanes')]\
            - destination_features[:, all_node_features.columns.get_loc('lanes')]
        # 最小-最大归一化车道数之差
        lane_diff_min = np.min(lane_diff)
        lane_diff_max = np.max(lane_diff)
        lane_diff_normalized = (lane_diff - lane_diff_min) / (lane_diff_max - lane_diff_min)

        # 计算长度之差
        length_diff = source_features[:, all_node_features.columns.get_loc('length_id')] \
            - destination_features[:, all_node_features.columns.get_loc('length_id')]
        # 最小-最大归一化长度之差
        length_diff_min = np.min(length_diff)
        length_diff_max = np.max(length_diff)
        length_diff_normalized = (length_diff - length_diff_min) / (length_diff_max - length_diff_min)

        # 构建边特征矩阵
        edge_features_matrix = np.column_stack([
            oneway_values,  # Oneway 属性值
            lane_diff_normalized,  # 车道数差异归一化
            np.logical_or(source_features[:, all_node_features.columns.get_loc('bridge')],
                          destination_features[:, all_node_features.columns.get_loc('bridge')]),  # 桥梁标志位
            np.logical_or(source_features[:, all_node_features.columns.get_loc('tunnel')],
                          destination_features[:, all_node_features.columns.get_loc('tunnel')]),  # 隧道标志位
            (source_features[:, all_node_features.columns.get_loc('highway_id')] ==
             destination_features[:, all_node_features.columns.get_loc('highway_id')]),  # 高速公路类型是否相同
            length_diff_normalized,  # 长度差异归一化
        ])

        # 将特征矩阵转换为 PyTorch 张量
        features_tensor = torch.tensor(edge_features_matrix, dtype=torch.float32)

        # 定义一个线性层来处理路段属性
        attr_linear = nn.Linear(6, 128)

        # 应用线性层
        processed_features = attr_linear(features_tensor)
        # 将张量移到CPU上并转换为NumPy数组
        edge_features = processed_features.detach().cpu().numpy()

        # 保存边特征到文件
        np.save(f'./data//{prefix}+temp_edge_features.npy', edge_features)


class StaticDataset(Dataset):
    def __init__(self, data, pair_nodes, prefix, edge_features_df, mat_padding_value, data_padding_value, gps_max_len, route_max_len):
        # 仅包含gps轨迹和route轨迹，route中包含路段的特征
        # 不包含路段过去n个时间戳的流量数据
        self.data = data.reset_index(drop=True)
        # gps_length, df = split_connection_subseq(self.data, pair_nodes)
        # gps_length = pd.DataFrame(gps_length)
        # 将road_timestamp列转换为列表并存储
        # self.road_timestamps = df['road_timestamp'].tolist()

        gps_length = pd.read_pickle(f'./data//{prefix}+1w_split_gps_length.pkl')
        df = pd.read_pickle(f'./data//{prefix}+1w_split_df.pkl')
        with open(f'./data//{prefix}+road_timestamps.pkl', 'rb') as f:
            self.road_timestamps = pickle.load(f)

        self.gps_length = torch.tensor(gps_length.values, dtype=torch.int)
        df.reset_index(inplace=True, drop=True)
        # 替换data中与df同名列的值
        self.data.loc[:, ['speed', 'acceleration', 'opath_list', 'interval', 'cpath_list', 'road_timestamp', 'road_interval']] = df
        print(f'{prefix}路段已完成分割')
        # generate_temp_edge(df, prefix, edge_features_df)

        x_time_feature_padded, x_indo_features_padded, x_exo_features_padded, x_opath_padded, \
        y_time_feature_padded, y_indo_features_padded, y_opath_padded = prepare_gps_data(self.data[['opath_list', 'tm_list',
                                                                                'lng_list', 'lat_list',
                                                                                'speed', 'acceleration',
                                                                                'angle_delta', 'interval',
                                                               'dist']], mat_padding_value, data_padding_value, gps_max_len)
        self.gps_data = GPSData(x_time_feature_padded, x_indo_features_padded, x_exo_features_padded, x_opath_padded,
                                y_time_feature_padded, y_indo_features_padded, y_opath_padded)
        # self.gps_data.scale()

        # # gps 点的信息，从tm_list，traj_list，speed，acceleration，angle_delta，interval，dist生成，padding_value = 0
        # self.gps_data = gps_data  # gps point shape = (num_samples,gps_max_length,num_features)
        # # 表示gps点数据属于哪个路段，从opath_list生成，padding_value = num_nodes
        # self.gps_assign_mat = gps_assign_mat
        # shape = (num_samples,gps_max_length)[[5366., 5366., 5366.,  ..., 6450., 6450., 6450.],
        # [1281., 1281., 1281.,  ..., 6450., 6450., 6450.],

        # # todo 路段本身的属性特征怎么放进去
        # route_data, route_assign_mat = prepare_route_data(self.data[['cpath_list', 'road_timestamp','road_interval']],\
        #                                                   mat_padding_value, data_padding_value, route_max_len)
        #
        # # route对应的信息，从road_interval生成，padding_value = 0
        # self.route_data = route_data  # shape = (num_samples,route_max_length,3)
        #
        # # 表示路段的序列信息，从cpath_list生成，padding_value = num_nodes
        # self.route_assign_mat = route_assign_mat  # shape = (num_samples,route_max_length)

        graph_df = pd.read_csv(f'./data/{prefix}+temp_edge.csv')
        sources = graph_df.source.values
        destinations = graph_df.destination.values
        edge_idxs = graph_df.index.values
        labels = graph_df.label.values

        self.graph_data = GraphData(
            sources=sources,
            destinations=destinations,
            edge_idxs=edge_idxs,
            labels=labels
        )
        
        if prefix == 'train':        
            # Initialize training neighbor finder to retrieve temporal graph
            self.train_ngh_finder = get_neighbor_finder(self.graph_data, self.road_timestamps, uniform=True)
            # Initialize negative samplers. Set seeds for validation and testing so negatives are the same
            # across different runs
            self.train_rand_sampler = RandEdgeSampler(self.graph_data.sources, self.graph_data.destinations)       
            self.device = torch.device('cuda')
        else:
            self.train_rand_sampler = None
            # # Compute time statistics
            # self.mean_time_shift_src, self.std_time_shift_src, self.mean_time_shift_dst, self.std_time_shift_dst = \
            #     compute_time_statistics(self.graph_data.sources, self.graph_data.destinations,
            #                             self.graph_data.timestamps)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return (self.gps_data.x_time_features_list[idx], self.gps_data.x_indo_features_list[idx],
                self.gps_data.x_exo_features_list[idx], self.gps_data.x_opath_list[idx],
                self.gps_data.y_time_features_list[idx], self.gps_data.y_indo_features_list[idx],
                self.gps_data.y_opath_list[idx],
                self.road_timestamps[idx], self.gps_length[idx], idx)

    def inverse_endogenous_transform(self, data):
        return self.gps_data.scaler_endogenous.inverse_transform(data)

    def inverse_exogenous_transform(self, data):
        return self.gps_data.scaler_exogenous.inverse_transform(data)

    def custom_collate_fn(self, batch):
        # 首先，将batch转换为易于处理的格式，例如一个列表或字典
        # 这里假设batch是一个列表，每个元素是一个样本
        batch_size = len(batch)
        x_time_features = torch.stack([item[0] for item in batch])
        x_endogenous_data = torch.stack([item[1] for item in batch])
        x_exogenous_data = torch.stack([item[2] for item in batch])
        x_opath_list_tensors = torch.stack([item[3] for item in batch])
        y_time_features = torch.stack([item[4] for item in batch])
        y_endogenous_data = torch.stack([item[5] for item in batch])
        y_opath_list_tensors = torch.stack([item[6] for item in batch])
        road_timestamps = [item[7] for item in batch]
        gps_length = [item[-2] for item in batch]

        # 然后，使用graph_data和train_rand_sampler处理数据
        indices = [item[-1] for item in batch]
        start_label = indices[0]
        end_label = min(len(self.graph_data.labels), start_label + batch_size)
        current_batch_labels = set(range(start_label, end_label))

        # 确定哪些索引对应的 label 在当前 batch 内
        indices_in_batch = [i for i, label in enumerate(self.graph_data.labels) if label in current_batch_labels]

        # 根据找到的索引提取 edge_idxs
        edge_idxs_batch = self.graph_data.edge_idxs[indices_in_batch]
        sources_batch, destinations_batch = self.graph_data.sources[indices_in_batch],\
                                            self.graph_data.destinations[indices_in_batch]

        if self.train_rand_sampler is not None:
            size = len(sources_batch)
            _, negatives_batch = self.train_rand_sampler.sample(size)
            with torch.no_grad():
                pos_label = torch.ones(size, dtype=torch.float, device=self.device)
                neg_label = torch.zeros(size, dtype=torch.float, device=self.device)
        else:
            negatives_batch = None
            pos_label = None
            neg_label = None

        # 最后，返回一个批次的数据
        # 这里返回的是一个字典，但你可以根据需要返回任何格式
        return {
            'x_indo_features': x_endogenous_data,
            'x_exo_features': x_exogenous_data,
            'x_opath': x_opath_list_tensors,
            'x_time_feature': x_time_features,
            'y_indo_features': y_endogenous_data,
            'y_opath': y_opath_list_tensors,
            'y_time_feature': y_time_features,
            'road_timestamps': road_timestamps,
            'gps_length': gps_length,
            'sources_batch': sources_batch,
            'destinations_batch': destinations_batch,
            'edge_idxs_batch': edge_idxs_batch,
            'negatives_batch': negatives_batch,
            'pos_label': pos_label,
            'neg_label': neg_label
        }



# class DynamicDataset_(Dataset):
#     def __init__(self, data, edge_features, traffic_flow_history):
#         # 仅包含gps轨迹和route轨迹，route中包含路段的特征
#         # 不包含路段过去n个时间戳的流量数据
#         self.data = data
#
#         # 特征矩阵和指示矩阵
#         # gps 点的信息，从tm_list，traj_list，speed，acceleration，angle_delta，interval，dist生成，padding_value = 0
#         self.gps_data = # gps point shape = (num_samples,gps_max_length,features)
#
#         # 表示gps点数据属于哪个路段，从opath_list生成，padding_value = num_nodes
#         self.gps_assign_mat = #  shape = (num_samples,gps_max_length)
#
#         # # route features
#         # # cpath_list
#         # # road_interval
#         # # 关联道路特征图有其他特征
#
#         # route对应的信息，从road_interval 和 edge_features生成，包含静态特征和动态特征，静态特征为路段特征，动态特征为历史流量信息，padding_value = 0
#         self.route_data = # shape = (num_samples,route_max_length,num_features)
#         self.route_assign_mat = # shape = (num_samples,route_max_length)
#
#         self.data_lst = []
#         self.data.groupby('pairid').apply(lambda group: self.data_lst.append(torch.tensor(np.array(group.iloc[:,:-2]))))
#         self.x_mbr = pad_sequence(self.data_lst, batch_first=True) # set_count*max_len*fea_dim
#         self.x_c = self.x_mbr
#         self.label_idxs = self.data.groupby('pairid').apply(lambda group:find_label_idx(group)).values # label=1 ndarray
#         self.lengths = self.data.groupby('pairid')['label'].count().values # ndarray
#
#     def __len__(self):
#         return len(self.data_lst)
#
#     def __getitem__(self, idx):
#         return (self.x_mbr[idx],self.x_c[idx],self.label_idxs[idx],self.lengths[idx])


class IndexSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.indices = list(range(len(data_source)))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.data_source)


def find_unique_edges(edge_index):
    # 初始化一个字典来记录每个节点出现的次数
    node_counts = {}
    # 遍历第一行的元素
    for node in edge_index[0]:
        # 如果节点不在字典中，添加它并设置计数为1
        # 如果节点已经在字典中，增加其计数
        if node.item() not in node_counts:
            node_counts[node.item()] = 1
        else:
            node_counts[node.item()] += 1

    # 创建一个列表来存储只出现一次的节点
    unique_nodes = []

    # 遍历字典，找出计数为1的节点
    for node, count in node_counts.items():
        if count == 1:
            unique_nodes.append(node)

    return unique_nodes


def get_loader(data_path, edge_index,edge_features,batch_size, num_worker, route_min_len, route_max_len, gps_min_len, gps_max_len, num_samples, seed, vocab_size, road_embed_size, init_road_emb, city):

    dataset = pickle.load(open(data_path, 'rb'))
    print(dataset.columns)

    dataset['route_length'] = dataset['cpath_list'].map(len)
    dataset = dataset[
        (dataset['route_length'] > route_min_len) & (dataset['route_length'] < route_max_len)].reset_index(drop=True)

    dataset['gps_length'] = dataset['opath_list'].map(len)
    dataset = dataset[
        (dataset['gps_length'] > gps_min_len) & (dataset['gps_length'] < gps_max_len)].reset_index(drop=True)
    dataset.sort_values(by='start_time', inplace=True)

    print(dataset.shape)
    # assert dataset.shape[0] >= num_samples

    # 获取最大路段id
    uniuqe_path_list = []
    dataset['cpath_list'].apply(lambda cpath_list: uniuqe_path_list.extend(list(set(cpath_list))))
    uniuqe_path_list = list(set(uniuqe_path_list))
    mat_padding_value = -1
    data_padding_value = 0.0

    dataset = dataset.reset_index(drop=True)
    # unique_nodes = find_unique_edges(edge_index)
    # # 只保留第1到第3列和第6到第9列
    # columns_to_keep = [0, 1, 2, 5, 6, 7, 8]
    # edge_features = edge_features.iloc[:, columns_to_keep]
    # # 将缺失值填充为0
    # edge_features.fillna(0, inplace=True)
    # # 归一化处理
    # scaler = MinMaxScaler()
    # numeric_columns = ['oneway', 'lanes', 'bridge', 'tunnel', 'highway_id', 'length_id']
    # edge_features[numeric_columns] = scaler.fit_transform(edge_features[numeric_columns])
    # # 打印处理后的 DataFrame 查看结果
    # print(edge_features.head())
    #
    # # 创建NodeEmbeddingModel实例
    # node_embedding_model = NodeEmbeddingModel(vocab_size=vocab_size, road_embed_size=road_embed_size,
    #                                           init_road_emb=init_road_emb, edge_features_df=edge_features)
    # # 如果需要在初始化时调用add_new_nodes或其他方法

    with open(r'./data//1w_pair_nodes.pkl', 'rb') as f:
        pair_nodes = pickle.load(f)
    # 加载处理过的node_embedding状态
    pretrained_embedding = torch.load(
        (r'./data/\1w_node_embedding.pth'.format(city)))
    pretrained_weights = pretrained_embedding['node_embedding.weight']
    vocab_size = pretrained_weights.shape[0]
    # 创建一个Embedding层，并加载预训练的权重
    embedding_layer = torch.nn.Embedding.from_pretrained(pretrained_weights)
    # 使用这个Embedding层创建NodeEmbeddingModel实例
    node_embedding_model = NodeEmbeddingModel(embedding_layer=embedding_layer)
    node_features = node_embedding_model.node_embedding.weight.detach()
    node_features.requires_grad_(True)
    prefixes = ['train', 'val', 'test']

    # dataset['flag'] = pd.to_datetime(dataset['start_time'], unit='s').dt.day
    # 前13天作为训练集，第14天作为测试集，第15天作为验证集
    # train_data, test_data, val_data =dataset[dataset['flag']<8], dataset[dataset['flag']==8], dataset[dataset['flag']==9]
    # # train_data = train_data.sample(n=num_samples, replace=False, random_state=seed)   
    
    # 计算分割点
    total_length = len(dataset)
    val_split = int(total_length * 0.70)  # 70% 的位置
    test_split = int(total_length * 0.85)  # 85% 的位置
    train_dataset = dataset.iloc[:val_split]  # 前 70%
    val_dataset = dataset.iloc[val_split:test_split]  # 接下来的 15%
    test_dataset = dataset.iloc[test_split:]  # 最后的 15%  
    
    # notice: 一般情况下
    train_dataset = StaticDataset(train_dataset, pair_nodes, 'train', edge_features, mat_padding_value, data_padding_value,gps_max_len,route_max_len)
    val_dataset = StaticDataset(val_dataset, pair_nodes, 'val', edge_features, mat_padding_value, data_padding_value,gps_max_len,route_max_len)
    test_dataset = StaticDataset(test_dataset, pair_nodes, 'test', edge_features, mat_padding_value, data_padding_value,gps_max_len,route_max_len)
    index_sampler = IndexSampler(train_dataset)

    # 定义文件路径
    path_prefix = './data'
    # 初始化一个空的 DataFrame 用于存储所有的数据
    all_graph_df = pd.DataFrame()
    all_timestamps=[]
    # 遍历每一个前缀
    for prefix in prefixes:
        # 构造完整的文件路径
        file_path1 = path_prefix + prefix + '+temp_edge.csv'
        file_path2 = path_prefix + prefix + '+road_timestamps.pkl'
        # 使用 glob 来查找文件，这里假设每次只会找到一个文件
        # 使用 glob 来查找文件
        files = glob(file_path1)
        pkl_files = glob(file_path2)

        # 如果找到了文件，则读取它
        if files:
            file = files[0]
            graph_df = pd.read_csv(file)
            # 将读取的 DataFrame 添加到总的数据集中
            all_graph_df = pd.concat([all_graph_df, graph_df], ignore_index=True)

        if pkl_files:
            file = pkl_files[0]
            with open(file, 'rb') as f:
                timestamps = pickle.load(f)
                all_timestamps.extend(timestamps)
    # 提取所需的数据
    sources = all_graph_df.source.values
    destinations = all_graph_df.destination.values
    edge_idxs = all_graph_df.index.values
    labels = all_graph_df.label.values
    graph_full_data = GraphData(sources=sources,
                                destinations=destinations,
                                edge_idxs=edge_idxs,
                                labels=labels)
    print("All data has been concatenated and the GraphData object is created.")
    # Initialize validation and test neighbor finder to retrieve temporal graph
    full_ngh_finder = get_neighbor_finder(graph_full_data, all_timestamps, uniform=True)
    # Initialize negative samplers. Set seeds for validation and testing so negatives are the same
    # across different runs
    val_rand_sampler = RandEdgeSampler(graph_full_data.sources, graph_full_data.destinations, seed=0)
    test_rand_sampler = RandEdgeSampler(graph_full_data.sources, graph_full_data.destinations, seed=2)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              collate_fn=train_dataset.custom_collate_fn,
                              sampler=index_sampler,
                              shuffle=False,
                              drop_last=True, num_workers=num_worker)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            collate_fn=val_dataset.custom_collate_fn,
                            shuffle=False,
                            drop_last=True, num_workers=num_worker)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             collate_fn=test_dataset.custom_collate_fn,
                             shuffle=False,
                             drop_last=True, num_workers=num_worker)

    return (
        train_loader,
        val_loader,
        test_loader,
        vocab_size,
        node_features,
        full_ngh_finder,
        train_dataset.train_rand_sampler,
        train_dataset.train_ngh_finder,
        test_rand_sampler,
        val_rand_sampler,
        train_dataset.device,
    )


def get_train_loader(data_path, edge_index,edge_features,batch_size, num_worker, route_min_len, route_max_len, gps_min_len, gps_max_len, num_samples, seed, vocab_size, road_embed_size, init_road_emb, city):

    dataset = pickle.load(open(data_path, 'rb'))
    print(dataset.columns)

    dataset['route_length'] = dataset['cpath_list'].map(len)
    dataset = dataset[
        (dataset['route_length'] > route_min_len) & (dataset['route_length'] < route_max_len)].reset_index(drop=True)

    dataset['gps_length'] = dataset['opath_list'].map(len)
    dataset = dataset[
        (dataset['gps_length'] > gps_min_len) & (dataset['gps_length'] < gps_max_len)].reset_index(drop=True)

    print(dataset.shape)
    print(num_samples)
    assert dataset.shape[0] >= num_samples

    # 获取最大路段id
    uniuqe_path_list = []
    dataset['cpath_list'].apply(lambda cpath_list: uniuqe_path_list.extend(list(set(cpath_list))))
    uniuqe_path_list = list(set(uniuqe_path_list))

    mat_padding_value = max(uniuqe_path_list) + 1
    data_padding_value = 0.0

    # 前13天作为训练集，第14天作为测试集，第15天作为验证集，已经提前分好
    train_data = dataset
    train_data = train_data.sample(n=num_samples, replace=False, random_state=seed)

    train_dataset = StaticDataset(train_data,edge_index,edge_features, mat_padding_value, data_padding_value,gps_max_len,route_max_len, vocab_size, road_embed_size, init_road_emb, city)
    # 创建自定义的sampler
    index_sampler = IndexSampler(train_dataset)

    # 现在在创建DataLoader时传入collate_fn
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              collate_fn=train_dataset.custom_collate_fn,
                              sampler=index_sampler,
                              shuffle=False,
                              drop_last=True, num_workers=num_worker)

    return (
        train_loader,
        train_dataset.vocab_size,
        train_dataset.node_features,
        train_dataset.train_ngh_finder,
        train_dataset.train_rand_sampler,
        train_dataset.device,
        train_dataset.mean_time_shift_src,
        train_dataset.std_time_shift_src,
        train_dataset.mean_time_shift_dst,
        train_dataset.std_time_shift_dst
    )


def random_mask(gps_assign_mat, route_assign_mat, gps_length, mask_token, mask_length=1, mask_prob=0.2):

    # mask route
    col_num = int(route_assign_mat.shape[1] / mask_length) + 1
    batch_size = route_assign_mat.shape[0]

    # mask的位置和padding的位置有重合，但整体mask概率无影响
    route_mask_pos = torch.empty(
        (batch_size, col_num),
        dtype=torch.float32,
        device=route_assign_mat.device).uniform_(0, 1) < mask_prob

    route_mask_pos = torch.stack(sum([[col]*mask_length for col in route_mask_pos.t()], []), dim=1)

    # 截断
    if route_mask_pos.shape[1] > route_assign_mat.shape[1]:
        route_mask_pos = route_mask_pos[:, :route_assign_mat.shape[1]]

    masked_route_assign_mat = route_assign_mat.clone()
    masked_route_assign_mat[route_mask_pos] = mask_token

    # mask gps
    masked_gps_assign_mat = gps_assign_mat.clone()
    gps_mask_pos = []
    for idx, row in enumerate(gps_assign_mat):
        route_mask = route_mask_pos[idx]
        length_list = gps_length[idx]
        unpad_mask_pos_list = sum([[mask] * length_list[_idx].item() for _idx, mask in enumerate(route_mask)], [])
        pad_mask_pos_list = unpad_mask_pos_list + [torch.tensor(False)] * (
                    gps_assign_mat.shape[1] - len(unpad_mask_pos_list))
        pad_mask_pos = torch.stack(pad_mask_pos_list)
        gps_mask_pos.append(pad_mask_pos)
    gps_mask_pos = torch.stack(gps_mask_pos, dim=0)
    masked_gps_assign_mat[gps_mask_pos] = mask_token
    # 获得每个gps点对应路段的长度

    return masked_route_assign_mat, masked_gps_assign_mat


class GraphData:
    def __init__(self, sources, destinations, edge_idxs, labels):
        self.device = 'cuda'
        self.sources = sources
        self.destinations = destinations
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)


TGNConfig = namedtuple('TGNConfig', [
    'neighbor_finder', 'node_features', 'edge_features', 'device', 'n_layers',
    'n_heads', 'dropout', 'use_memory', 'message_dimension', 'memory_dimension',
    'memory_update_at_start', 'embedding_module_type', 'message_function',
    'aggregator_type', 'memory_updater_type', 'n_neighbors',
    'use_destination_embedding_in_message', 'use_source_embedding_in_message',
    'dyrep'
])
GPSConfig = namedtuple('GPSConfig', [
    'model_name', 'seq_len', 'pred_len', 'enc_in_endo', 'enc_in_exo', 'e_layers', 'n_heads', 'd_model', 'd_ff', 'dropout', 'fc_dropout',
    'head_dropout', 'individual', 'patch_len', 'stride', 'padding_patch', 'do_predict', 'revin', 'affine', 'subtract_last',
    'decomposition', 'kernel_size'
])


class GPSData:
    def __init__(self, x_time_feature_padded, x_indo_features_padded, x_exo_features_padded, x_opath_padded,
                 y_time_feature_padded, y_indo_features_padded, y_opath_padded):
        self.x_time_features_list = x_time_feature_padded
        self.x_indo_features_list = x_indo_features_padded
        self.x_exo_features_list = x_exo_features_padded
        self.x_opath_list = x_opath_padded

        self.y_time_features_list = y_time_feature_padded
        self.y_indo_features_list = y_indo_features_padded
        self.y_opath_list = y_opath_padded

        self.scaler_endogenous = StandardScaler()
        self.scaler_exogenous = StandardScaler()

    def scale(self):
        # Fit scaler to the data
        x_indo_np = self.x_indo_features_list.numpy()
        y_indo_np = self.y_indo_features_list.numpy()
        # 将张量重塑为二维的，以便可以进行fit和transform
        xy_indo_features = np.concatenate((x_indo_np, y_indo_np), axis=1)
        # 重塑 indo 张量为二维形式，以便可以使用 StandardScaler
        xy_indo_reshaped = xy_indo_features.reshape(-1, xy_indo_features.shape[-1])
        # 初始化并拟合 scaler
        self.scaler_endogenous.fit(xy_indo_reshaped)
        # 使用拟合后的 scaler 进行转换
        xy_indo_scaled = self.scaler_endogenous.transform(xy_indo_reshaped)
        # 将转换后的 indo 数据恢复为原来的形状
        xy_indo_scaled = xy_indo_scaled.reshape(xy_indo_features.shape)
        # 分离出原始的 indo 张量
        x_indo_scaled = xy_indo_scaled[:, :self.x_indo_features_list.shape[1]]
        y_indo_scaled = xy_indo_scaled[:, self.x_indo_features_list.shape[1]:]

        x_exo_reshaped = self.x_exo_features_list.reshape(-1, self.x_exo_features_list.shape[-1])
        self.scaler_exogenous.fit(x_exo_reshaped)
        # 使用拟合后的scaler进行转换
        x_exo_scaled = self.scaler_exogenous.transform(x_exo_reshaped)

        # 将转换后的数据恢复为原来的形状
        # 将标准化后的 NumPy 数组转换回张量
        x_indo_scaled = torch.from_numpy(x_indo_scaled)
        y_indo_scaled = torch.from_numpy(y_indo_scaled)
        x_exo_scaled = torch.from_numpy(x_exo_scaled.reshape(self.x_exo_features_list.shape))

        self.x_indo_features_list = x_indo_scaled
        self.y_indo_features_list = y_indo_scaled
        self.x_exo_features_list = x_exo_scaled


def get_middle_timestamps_datetime(timestamps):
    middle_timestamps = []
    deltas = []
    for i in range(len(timestamps) - 1):
        delta = timestamps[i+1] - timestamps[i]
        middle_time = timestamps[i] + (delta / 2)
        middle_timestamps.append(middle_time)
        deltas.append(delta)
    return middle_timestamps, deltas
