import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import double
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GATConv
from basemodel import BaseModel
import torch.nn.utils.rnn as rnn_utils
from dataloader import GraphData
from graph_utils.data_processing import compute_time_statistics
from graph_utils.utils import get_neighbor_finder, RandEdgeSampler
from model.tgn import TGN
from split_subseq import reverse_decimal_to_integer
from gps_model import Transformer, Informer, Reformer, Flowformer, Flashformer, \
    iTransformer, iInformer, iReformer, iFlowformer, iFlashformer, TimeXer, PatchTST


class DTSModel(BaseModel):
    def __init__(self, vocab_size, route_max_len, road_feat_num, road_embed_size, gps_feat_num, gps_embed_size,
                 route_embed_size, hidden_size, edge_index, drop_edge_rate, drop_route_rate, drop_road_rate, nodes_dict,
                 edge_features_df, tgn_config, gps_config, mode='p'):
        super(DTSModel, self).__init__()

        self.new_vocab_size = vocab_size  # 路段数量
        self.vocab_size = vocab_size+1
        self.edge_index = torch.tensor(edge_index).cuda()
        self.mode = mode
        self.drop_edge_rate = drop_edge_rate
        self.nodes_dict = nodes_dict
        self.edge_features_df = edge_features_df

        # Create heterogeneous graph
        # Define node and edge types
        self.node_types = ['original', 'new']
        self.edge_types = [('original', 'connects', 'original'), ('original', 'connects', 'new'),
                           ('new', 'connects', 'original')]
        self.create_hetero_graph()

        # node embedding
        self.route_padding_vec = torch.zeros(1, road_embed_size, requires_grad=True).cuda()

        # time embedding 考虑加法, 保证 embedding size一致
        self.minute_embedding = nn.Embedding(1440 + 1, route_embed_size)    # 0 是mask位
        self.week_embedding = nn.Embedding(7 + 1, route_embed_size)         # 0 是mask位
        self.delta_embedding = IntervalEmbedding(100, route_embed_size)     # -1 是mask位

        # route encoding
        self.graph_encoder = GraphEncoder(road_embed_size, route_embed_size)
        self.position_embedding1 = nn.Embedding(route_max_len, route_embed_size)
        self.fc1 = nn.Linear(route_embed_size, hidden_size)  # route fuse time ffn
        self.route_encoder = TransformerModel(hidden_size, 8, hidden_size, 4, drop_route_rate)
        self.tgn = TGN(
            neighbor_finder=tgn_config.neighbor_finder, node_features=tgn_config.node_features,
            edge_features=tgn_config.edge_features, device=tgn_config.device,
            n_layers=tgn_config.n_layers, n_heads=tgn_config.n_heads, dropout=tgn_config.dropout,
            use_memory=tgn_config.use_memory, message_dimension=tgn_config.message_dimension,
            memory_dimension=tgn_config.memory_dimension, memory_update_at_start=tgn_config.memory_update_at_start,
            embedding_module_type=tgn_config.embedding_module_type, message_function=tgn_config.message_function,
            aggregator_type=tgn_config.aggregator_type, memory_updater_type=tgn_config.memory_updater_type,
            n_neighbors=tgn_config.n_neighbors,
            use_destination_embedding_in_message=tgn_config.use_destination_embedding_in_message,
            use_source_embedding_in_message=tgn_config.use_source_embedding_in_message,
            dyrep=tgn_config.dyrep
        )

        self.gps_model_dict = {
            'Transformer': Transformer,
            'Informer': Informer,
            'Reformer': Reformer,
            'Flowformer': Flowformer,
            'Flashformer': Flashformer,
            'iTransformer': iTransformer,
            'iInformer': iInformer,
            'iReformer': iReformer,
            'iFlowformer': iFlowformer,
            'iFlashformer': iFlashformer,
            'TimeXer': TimeXer,
            'PatchTST': PatchTST
        }
        self.pre_len = gps_config.pred_len
        self.gps_model = self.gps_model_dict[gps_config.model_name].Model(gps_config).float().to('cuda:0')
        self.EuclidistanceLoss = nn.HuberLoss(delta=1.0)

        # gps encoding
        self.gps_linear = nn.Linear(gps_feat_num, gps_embed_size)
        self.gps_intra_encoder = nn.GRU(gps_embed_size, gps_embed_size, bidirectional=True, batch_first=True) # 路段内建模
        self.gps_inter_encoder = nn.GRU(gps_embed_size, gps_embed_size, bidirectional=True, batch_first=True) # 路段间建模

        # cl project head
        self.gps_proj_head = nn.Linear(2*gps_embed_size, hidden_size)
        self.route_proj_head = nn.Linear(hidden_size, hidden_size)

        # shared transformer
        self.position_embedding2 = nn.Embedding(route_max_len, hidden_size)
        self.modal_embedding = nn.Embedding(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size) # shared transformer position transform
        self.sharedtransformer = TransformerModel(hidden_size, 4, hidden_size, 2, drop_road_rate)

        # mlm classifier head
        self.gps_mlm_head = nn.Linear(hidden_size, vocab_size)
        self.route_mlm_head = nn.Linear(hidden_size, vocab_size)

        # matching
        self.matching_predictor = nn.Linear(hidden_size, 2)
        self.register_buffer("gps_queue", torch.randn(hidden_size, 2048))
        self.register_buffer("route_queue", torch.randn(hidden_size, 2048))

        self.image_queue = nn.functional.normalize(self.gps_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.route_queue, dim=0)

    def create_hetero_graph(self):
        # Define the edges for each type separately
        # Assuming that edge_index already contains the edges between original nodes
        original_edges = (self.edge_index[0].tolist(), self.edge_index[1].tolist())

        # Initially, there are no edges between new nodes or between original and new nodes
        original_new_edges = ([], [])
        new_original_edges = ([], [])
        new_new_edges = ([], [])

    def encode_graph(self, drop_rate=0.):
        node_emb = self.node_embedding.weight
        edge_index = dropout_adj(self.edge_index, p=drop_rate)[0]
        node_enc = self.graph_encoder(node_emb, edge_index)
        return node_enc

    def encode_route(self, sources_batch, destinations_batch, negatives_batch,
                     src_timestamps_batch, des_timestamps_batch, route_len, intervals, edge_idxs_batch):

        pos_prob, neg_prob = self.tgn.compute_edge_probabilities(sources_batch, destinations_batch, negatives_batch,
                                                                 src_timestamps_batch, des_timestamps_batch, route_len, intervals,
                                                                 edge_idxs_batch, n_neighbors=self.tgn.n_neighbors)
        return pos_prob, neg_prob

    def encode_gps(self, x_indo_features_batch, x_exo_features_batch, x_opath_batch, x_time_feature_batch,
                   y_indo_features_batch, y_opath_batch, y_time_feature_batch, gps_length):

        outputs = self.gps_model(x_exo_features_batch, x_indo_features_batch, x_time_feature_batch, x_opath_batch, y_opath_batch, self.pre_len)
        f_dim = -1
        outputs = outputs[:, -self.pre_len:, f_dim:]
        seq_endogenous_y = y_indo_features_batch[0].to('cuda:0')  # 对于TimeXer目前只有一个变量，所以f_dim是啥都无所谓
        return outputs, seq_endogenous_y

    def route_stack(self, gps_emb, route_length):
        # flatten_gps_data tensor = (965,128)
        # stacked_gps_emb = (batch_size, max_subsequence_in_route_len, emb_size)
        # route_length dict = { key:tid, value: road_len }
        values = list(route_length.values())  # batch中每条轨迹包含的非0路段(subsequence)的个数
        route_max_len = max(values)
        data_list = []
        for idx in range(len(route_length)):
            start_idx = sum(values[:idx])
            end_idx = sum(values[:idx+1])
            data = gps_emb[start_idx:end_idx]
            data_list.append(data)

        stacked_gps_emb = rnn_utils.pad_sequence(data_list, padding_value=0, batch_first=True)

        return stacked_gps_emb

    def gps_flatten(self, gps_data, gps_length):
        # 把gps_data按照gps_assign_mat做形变，把每个路段上的gps点单独拿出来，拼成一个新的tensor (road_num, gps_max_len, gps_feat_num)，
        # 该tensor用于输入GRU进行并行计算
        traj_num, gps_max_len, gps_feat_num = gps_data.shape
        flattened_gps_list = []
        route_index = {}
        for idx in range(traj_num):
            gps_feat = gps_data[idx] # (max_len, feat_num)
            length_list = gps_length[idx] # (max_len, 1) [7,9,12,1,0,0,0,0,0,0] # padding_value = 0
            # 遍历每个轨迹中的路段，得到batch中所有subsequence即同路段gps点组，每个gps点有128个特征
            for _idx, length in enumerate(length_list):
                if length != 0:
                    start_idx = sum(length_list[:_idx])
                    end_idx = start_idx + length_list[_idx]
                    cnt = route_index.get(idx, 0)
                    route_index[idx] = cnt+1  # 统计每个轨迹中不同路段的个数
                    road_feat = gps_feat[start_idx:end_idx]
                    flattened_gps_list.append(road_feat)
        if flattened_gps_list:
            flattened_gps_data = rnn_utils.pad_sequence(flattened_gps_list, padding_value=0, batch_first=True)
        else:
            print("Warning: No valid GPS sequences found. Returning a placeholder tensor.")
            # 根据你的需求决定返回什么，这里以全零张量为例
            flattened_gps_data = torch.zeros(1, 1, gps_feat_num)  # 一个占位符，实际使用中可能需要调整大小

        return flattened_gps_data, route_index

    def encode_joint(self, route_road_rep, route_traj_rep, gps_road_rep, gps_traj_rep, route_assign_mat):
        max_len = torch.max((route_assign_mat != self.vocab_size).int().sum(1)).item()
        max_len = max_len*2+2
        data_list = []
        mask_list = []
        route_length = [length[length != self.vocab_size].shape[0] for length in route_assign_mat]

        modal_emb0 = self.modal_embedding(torch.tensor(0).cuda())
        modal_emb1 = self.modal_embedding(torch.tensor(1).cuda())

        for i, length in enumerate(route_length):
            route_road_token = route_road_rep[i][:length]
            gps_road_token = gps_road_rep[i][:length]
            route_cls_token = route_traj_rep[i].unsqueeze(0)  # 起始位
            gps_cls_token = gps_traj_rep[i].unsqueeze(0)

            # position
            position = torch.arange(length+1).long().cuda()
            pos_emb = self.position_embedding2(position)

            # update route_emb
            route_emb = torch.cat([route_cls_token, route_road_token], dim=0)
            modal_emb = modal_emb0.unsqueeze(0).repeat(length+1, 1)
            route_emb = route_emb + pos_emb + modal_emb
            route_emb = self.fc2(route_emb)  # 将这三者相加(route_emb + pos_emb + modal_emb)是为了综合考虑路线的特征、顺序信息及输入数据的模态特性，生成更加丰富的特征表示，
                                              # 供给后续的网络层(self.fc2)进一步处理和学习更高层次的抽象特征。如果没有self.fc2，模型将无法对综合了位置和模态信息的route_emb进行进一步的特征提取和变换，可能会限制模型的学习能力和性能。

            # update gps_emb
            gps_emb = torch.cat([gps_cls_token, gps_road_token], dim=0)
            modal_emb = modal_emb1.unsqueeze(0).repeat(length+1, 1)
            gps_emb = gps_emb + pos_emb + modal_emb
            gps_emb = self.fc2(gps_emb)

            data = torch.cat([gps_emb, route_emb], dim=0)  # 维度变为两倍
            data_list.append(data)

            mask = torch.tensor([False] * data.shape[0]).cuda()  # mask的位置为true
            mask_list.append(mask)

        joint_data = rnn_utils.pad_sequence(data_list, padding_value=0, batch_first=True)
        mask_mat = rnn_utils.pad_sequence(mask_list, padding_value=True, batch_first=True)

        joint_emb = self.sharedtransformer(joint_data, None, mask_mat)  # 作用于gps_traj、gps_road、route_traj、route_road，要改
        # 每一行的0 和 length+1 对应的是 gps_traj_rep 和 route_traj_rep
        # 以下四者互不重复
        gps_traj_rep = joint_emb[:, 0]
        route_traj_rep = torch.stack([joint_emb[i, length+1] for i, length in enumerate(route_length)], dim=0)

        gps_road_rep = rnn_utils.pad_sequence([joint_emb[i, 1:length+1] for i, length in enumerate(route_length)],
                                              padding_value=0, batch_first=True)
        route_road_rep = rnn_utils.pad_sequence([joint_emb[i, length+2:2*length+2] for i, length in enumerate(route_length)],
                                                padding_value=0, batch_first=True)

        return gps_road_rep, gps_traj_rep, route_road_rep, route_traj_rep

    def record_previous_paths(self, x_opath_batch, x_indo_features_batch,
                              x_time_feature_batch_np, y_time_feature_batch_np,  y_opath_batch, outputs_np):
        # 应用函数到 x_opath_batch 的每一行
        end_indices = np.apply_along_axis(find_last_group_indices, 1, x_opath_batch)

        # 初始化一个空的二维数组来存储结果，形状为 (batch_size, max_num_points_in_group)
        max_num_points_in_group = max([end - start + 1 for start, end in end_indices if start != -1])
        last_valid_path_ids = np.full((x_opath_batch.shape[0], max_num_points_in_group), -1, dtype=np.float64)
        last_valid_speeds = np.full((x_opath_batch.shape[0], max_num_points_in_group), -1, dtype=np.float32)
        last_valid_timestamps = np.full((x_opath_batch.shape[0], max_num_points_in_group), -1, dtype=np.long)

        # 提取对应的路段编号并填充到 last_valid_path_ids 中
        for i, (start, end) in enumerate(end_indices):
            if start != -1:
                valid_path_ids = x_opath_batch[i, start:end + 1]  # 提取这一段的所有值
                valid_speeds = x_indo_features_batch[i, start:end + 1, 0]
                valid_timestamps = x_time_feature_batch_np[i, start:end + 1]
                last_valid_path_ids[i, -len(valid_path_ids):] = valid_path_ids
                last_valid_speeds[i, -len(valid_speeds):] = valid_speeds
                last_valid_timestamps[i, -len(valid_timestamps):] = valid_timestamps

        # 拼接结果
        y_opath_batch = np.hstack((last_valid_path_ids, y_opath_batch))
        outputs_np = np.hstack((last_valid_speeds, outputs_np.squeeze(-1)))
        truth_timestamps_np = np.hstack((last_valid_timestamps, y_time_feature_batch_np))

        unique_path_id = []
        time = []
        truth_interval = []
        for m, sequence in enumerate(y_opath_batch):
            path_ids = sequence  # 确保可以比较浮点数
            skip_next = False  # 标志位，用于控制是否跳过下一次循环

            # 使用 NumPy 查找连续的路段编号
            unique_ids, starts, counts = np.unique(path_ids, return_counts=True, return_index=True)
            sorted_indices = np.argsort(starts)
            ends = starts + counts
            # 根据排序后的索引重新排列 unique_ids 和 counts
            starts = starts[sorted_indices]
            unique_ids = unique_ids[sorted_indices]
            ends = ends[sorted_indices]
            # 检查 counts 是否表示连续出现
            is_continuous = np.zeros_like(counts, dtype=bool)
            is_continuous[:-2] = (starts[2:] == ends[1:-1])
            is_continuous[-2:] = True  # 最后一个元素总是连续的，因为它没有后续元素来比较
            # 根据连续性进行处理
            if not is_continuous.all():
                # 重新构造 unique_ids、starts 和 ends，反映实际的连续段落
                new_unique_ids = []
                new_starts = []
                new_ends = []
                current_id = -1
                start_idx = None

                # 遍历路径 ID 并找出连续段
                for idx, id in enumerate(path_ids):
                    if current_id is -1:
                        current_id = id
                        start_idx = idx
                    elif id != current_id:
                        end_idx = idx - 1
                        new_unique_ids.append(current_id)
                        new_starts.append(start_idx)
                        new_ends.append(end_idx)
                        current_id = id
                        start_idx = idx

                # 添加最后一个段落的信息
                end_idx = len(path_ids)
                new_unique_ids.append(current_id)
                new_starts.append(start_idx)
                new_ends.append(end_idx)
                # 转换为 NumPy 数组
                unique_ids = np.array(new_unique_ids)
                starts = np.array(new_starts)
                ends = np.array(new_ends)

            # 倒序遍历每个唯一的路段编号及其对应的索引范围
            for i in reversed(range(len(unique_ids))):
                if skip_next:
                    skip_next = False  # 重置标志位
                    continue  # 跳过当前循环
                start, end, path_id = starts[i], ends[i], unique_ids[i]
                path_id_int = np.floor(path_id)
                if path_id != path_id_int and path_id != -1:
                    # 使用 get 方法，如果没有找到键，则返回默认值 None
                    node_inverted = float(self.nodes_dict.get(path_id, -1))
                    if node_inverted == -1:
                        print(f"没有找到节点信息: {path_id}")
                    # 查找前一个不同的路段编号
                    prev_id = unique_ids[i - 1] if i > 0 else None

                    if prev_id is not None:
                        prev_start, prev_end = starts[i - 1], ends[i - 1]
                        # 更新当前路段编号为前一个不同的路段编号
                        average_speed = np.mean(outputs_np[m, prev_start:end + 1])
                        interval = truth_timestamps_np[m, end-1].item() - truth_timestamps_np[m, start].item()
                        prev_interval = truth_timestamps_np[m, prev_end-1].item() - truth_timestamps_np[m, prev_start].item()
                        length = self.edge_features_df.loc[path_id_int]['length']
                        travel_time = length / np.abs(average_speed)
                        # 计算点数
                        point_count_prev = prev_end - prev_start + 1
                        point_count_curr = end - start + 1
                        # 计算比率
                        ratio1 = point_count_curr / (point_count_curr + point_count_prev)
                        time.extend([travel_time * ratio1, travel_time * (1 - ratio1)])
                        unique_path_id.extend((node_inverted, path_id_int))
                        truth_interval.extend([interval, prev_interval])
                    skip_next = True
                elif path_id == -1:
                    continue
                else:
                    # 对于其他情况，直接添加记录
                    average_speed = np.mean(outputs_np[m, start:end + 1])
                    interval = truth_timestamps_np[m, end-1].item() - truth_timestamps_np[m, start].item()
                    length = self.edge_features_df.loc[path_id]['length']
                    travel_time = length / np.abs(average_speed)
                    time.extend([travel_time])
                    unique_path_id.extend([path_id])
                    truth_interval.extend([interval])

        return time, unique_path_id, truth_interval

    def calculate_average_speeds(self, x_opath_batch, y_opath_batch, x_indo_features_batch,
                                 x_time_feature_batch, y_time_feature_batch, outputs):
        """
        计算每个路段编号对应的平均速度。

        参数:
        y_opath_batch (torch.Tensor): 形状为 (batch_size, num_points)，存储每个轨迹点所在的路段编号。
        outputs (torch.Tensor): 形状为 (batch_size, num_points, 1)，存储每个轨迹点的速度。

        返回:
        average_speeds (np.ndarray): 按路段编号顺序排列的平均速度数组。
        """
        # 只取第一个批次的数据作为示例
        y_opath_batch_np = y_opath_batch[0].cpu().numpy()
        x_opath_batch_np = x_opath_batch[0].cpu().numpy()
        x_indo_features_batch_np = x_indo_features_batch[0].cpu().numpy()
        x_time_feature_batch_np = x_time_feature_batch[0].cpu().numpy()
        y_time_feature_batch_np = y_time_feature_batch[0].cpu().numpy()
        outputs_np = outputs.cpu().detach().numpy()
        time, unique_path_id, truth_interval = self.record_previous_paths(x_opath_batch_np, x_indo_features_batch_np,
                                                          x_time_feature_batch_np, y_time_feature_batch_np,
                                                          y_opath_batch_np, outputs_np)
        return time, unique_path_id, truth_interval

    def forward(self, x_indo_features_batch, x_exo_features_batch, x_opath_batch, x_time_feature_batch,
                y_indo_features_batch, y_opath_batch, y_time_feature_batch, gps_length,
                sources_batch, destinations_batch, edge_idxs_batch,
                src_timestamps_batch,  des_timestamps_batch, route_len, intervals,
                negatives_batch):
        pos_prob, neg_prob = self.encode_route(sources_batch, destinations_batch, negatives_batch,
                                               src_timestamps_batch, des_timestamps_batch, route_len, intervals,
                                               edge_idxs_batch)

        outputs, seq_endogenous_y = self.encode_gps(x_indo_features_batch, x_exo_features_batch, x_opath_batch,
                                                    x_time_feature_batch, y_indo_features_batch, y_opath_batch,
                                                    y_time_feature_batch, gps_length)
        y_valid = torch.logical_not(torch.eq(y_opath_batch[0], -1.0))
        time, unique_path_id, truth_interval = self.calculate_average_speeds(x_opath_batch, y_opath_batch, x_indo_features_batch,
                                                             x_time_feature_batch, y_time_feature_batch, outputs)

        return pos_prob, neg_prob, outputs, seq_endogenous_y, y_valid, time, unique_path_id, truth_interval


# GAT
class GraphEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(GraphEncoder, self).__init__()
        # update road edge features using GAT
        self.layer1 = GATConv(input_size, output_size)
        self.layer2 = GATConv(input_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.activation(self.layer1(x, edge_index))
        x = self.activation(self.layer2(x, edge_index))
        return x


class TransformerModel(nn.Module):  # vanilla transformer
    def __init__(self, input_size, num_heads, hidden_size, num_layers, dropout=0.3):
        super(TransformerModel, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(input_size, num_heads, hidden_size, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, src, src_mask, src_key_padding_mask):
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        return output


# Continuous time embedding分箱编码
class IntervalEmbedding(nn.Module):
    def __init__(self, num_bins, hidden_size):
        super(IntervalEmbedding, self).__init__()
        self.layer1 = nn.Linear(1, num_bins)
        self.emb = nn.Embedding(num_bins, hidden_size)
        self.activation = nn.Softmax()

    def forward(self, x):
        logit = self.activation(self.layer1(x.unsqueeze(-1)))
        output = logit @ self.emb.weight
        return output


def find_last_group_indices(sequence):
    # 查找所有非 -1 值的位置
    non_minus_one_indices = np.where(sequence != -1)[0]
    last_int_indices = np.where((sequence != -1) & (sequence == np.floor(sequence)))[0][::-1][0]

    # 如果序列中没有任何非 -1 值，则返回 (-1, -1) 表示没有有效的路段编号
    if non_minus_one_indices.size == 0:
        return (-1, -1)
    # 获取最后一个非 -1 值的索引
    last_non_minus_one_indices = non_minus_one_indices[-1]

    # 查找该值之前最近的一个不同值的位置
    last_int_value = sequence[last_int_indices]
    reversed_seq = sequence[:last_int_indices][::-1]
    different_value_index = np.argmax(reversed_seq != last_int_value)

    # 如果找不到不同的值，说明整个序列都是同一个非 -1 值
    if different_value_index == 0 and reversed_seq[0] == -1:
        start_index = 0
    else:
        start_index = last_int_indices - different_value_index

    return (start_index, last_non_minus_one_indices)


def record_previous_paths(x_opath_batch, x_indo_features_batch, y_opath_batch, outputs_np):
    # 应用函数到 x_opath_batch 的每一行
    end_indices = np.apply_along_axis(find_last_group_indices, 1, x_opath_batch)

    # 初始化一个空的二维数组来存储结果，形状为 (batch_size, max_num_points_in_group)
    max_num_points_in_group = max([end - start + 1 for start, end in end_indices if start != -1])
    last_valid_path_ids = np.full((x_opath_batch.shape[0], max_num_points_in_group), -1, dtype=np.float64)
    last_valid_speeds = np.full((x_opath_batch.shape[0], max_num_points_in_group), -1, dtype=np.float32)

    # 提取对应的路段编号并填充到 last_valid_path_ids 中
    for i, (start, end) in enumerate(end_indices):
        if start != -1:
            valid_path_ids = x_opath_batch[i, start:end + 1]  # 提取这一段的所有值
            valid_speeds = x_indo_features_batch[i, start:end + 1, 0]
            last_valid_path_ids[i, -len(valid_path_ids):] = valid_path_ids
            last_valid_speeds[i, -len(valid_speeds):] = valid_speeds

    # 拼接结果
    y_opath_batch = np.hstack((last_valid_path_ids, y_opath_batch))
    outputs_np = np.hstack((last_valid_speeds, outputs_np.squeeze(-1)))

    records = []
    speeds = []
    path_ids = []
    for m, sequence in enumerate(y_opath_batch):
        path_ids = sequence  # 确保可以比较浮点数
        sequence_records = {}
        unique_speeds = []
        unique_path_id = []
        skip_next = False  # 标志位，用于控制是否跳过下一次循环

        # 使用 NumPy 查找连续的路段编号
        unique_ids, starts, counts = np.unique(path_ids, return_counts=True, return_index=True)
        # 计算每个唯一元素的结束索引
        ends = starts + counts
        # 检查 counts 是否表示连续出现
        is_continuous = np.zeros_like(counts, dtype=bool)
        is_continuous[:-1] = (starts[1:] == ends[:-1])
        is_continuous[-1] = True  # 最后一个元素总是连续的，因为它没有后续元素来比较
        # 根据连续性进行处理
        if is_continuous.all():
            # 根据 starts 排序
            sorted_indices = np.argsort(starts)
            # 根据排序后的索引重新排列 unique_ids 和 counts
            starts = starts[sorted_indices]
            unique_ids = unique_ids[sorted_indices]
            counts = counts[sorted_indices]

        # 倒序遍历每个唯一的路段编号及其对应的索引范围
        for i in reversed(range(len(unique_ids))):
            if skip_next:
                skip_next = False  # 重置标志位
                continue  # 跳过当前循环
            start, end, path_id = starts[i], ends[i], unique_ids[i]
            if path_id != np.floor(path_id) and path_id != -1:
                # 查找前一个不同的路段编号
                prev_id = unique_ids[i - 1] if i > 0 else None

                if prev_id is not None:
                    prev_start, prev_end = starts[i - 1], ends[i - 1]

                    # 更新当前路段编号为前一个不同的路段编号
                    average_speed = np.mean(outputs_np[m, prev_start:end + 1])
                    unique_speeds.extend(average_speed)
                    unique_path_id.extend(prev_id)
                    # 计算点数
                    point_count_prev = prev_end - prev_start + 1
                    point_count_curr = end - start + 1
                    # 计算比率
                    ratio1 = point_count_curr / (point_count_curr + point_count_prev)
                    # 添加记录
                    if prev_id not in sequence_records:
                        sequence_records[prev_id] = []
                    sequence_records[prev_id].extend([
                        path_id,  # 当前路段编号
                        ratio1
                    ])
                skip_next = True
            elif path_id == -1:
                continue
            else:
                # 对于其他情况，直接添加记录
                average_speed = np.mean(outputs_np[m, start:end + 1])
                unique_speeds.extend(average_speed)
                unique_path_id.extend(path_id)

        # 反转 unique_speeds 列表
        unique_speeds.reverse()
        unique_path_id.reverse()
        # 将每行的 speed 结果添加到 speeds 列表中
        speeds.extend(unique_speeds)
        path_ids.extend(unique_path_id)
        records.append(sequence_records)

    return records, path_ids, speeds


def calculate_average_speeds(x_opath_batch, y_opath_batch, x_indo_features_batch, outputs):
    """
    计算每个路段编号对应的平均速度。

    参数:
    y_opath_batch (torch.Tensor): 形状为 (batch_size, num_points)，存储每个轨迹点所在的路段编号。
    outputs (torch.Tensor): 形状为 (batch_size, num_points, 1)，存储每个轨迹点的速度。

    返回:
    average_speeds (np.ndarray): 按路段编号顺序排列的平均速度数组。
    """
    # 只取第一个批次的数据作为示例
    y_opath_batch_np = y_opath_batch[0].cpu().numpy()
    x_opath_batch_np = x_opath_batch[0].cpu().numpy()
    x_indo_features_batch_np = x_indo_features_batch[0].cpu().numpy()
    outputs_np = outputs.cpu().detach().numpy()
    records, path_ids, speeds = record_previous_paths(x_opath_batch_np, x_indo_features_batch_np, y_opath_batch_np, outputs_np)
    return path_ids, speeds, records

    # 展平 outputs 数组以便更容易地处理
    outputs_flat = outputs_np.flatten()
    # 初始化一个空列表来保存所有独特路段编号
    all_unique_path_ids = []

    for sample in y_opath_batch_np:
        # 找到该样本中的独特路段编号
        unique_path_ids = np.unique(sample)
        # 将这些独特路段编号添加到总的列表中
        all_unique_path_ids = np.concatenate((all_unique_path_ids, unique_path_ids))

    # 初始化一个张量来存储每个路段编号的平均速度
    average_speeds = torch.zeros((len(all_unique_path_ids), 1), dtype=torch.float32, device=y_opath_batch.device)
    if len(all_unique_path_ids) == len(set(all_unique_path_ids)):
        # 计算每个路段编号对应的平均速度
        for path_id in all_unique_path_ids:
            indices = np.where(y_opath_batch_np == path_id)
            speeds_on_path = outputs_flat[indices]
            if speeds_on_path.size > 0:
                average_speed = np.mean(speeds_on_path)
                index = np.where(all_unique_path_ids == path_id)[0][0]
                average_speeds[index] = torch.tensor([average_speed], dtype=torch.float32, device=y_opath_batch.device)
    else:
        # 计算每个路段编号对应的平均速度
        for path_id in all_unique_path_ids:
            indices = np.where(y_opath_batch_np.flatten() == path_id)[0]
            # 如果索引不为空
            if indices.size > 0:
                # 将索引转换为连续的段落
                segments = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
                path_average_speeds = []

                for i, segment in segments:
                    # 提取该路段上对应的速度值
                    speeds_on_path = outputs_flat[segment]
                    if speeds_on_path.size > 0:
                        average_speed = np.mean(speeds_on_path)
                        path_average_speeds.append(
                            torch.tensor([average_speed], dtype=torch.float32, device=y_opath_batch.device))

                # 将所有平均速度存储到 average_speeds 的相应位置
                index = np.where(all_unique_path_ids == path_id)[0][0]

                # 确保 path_average_speeds 有足够的元素来赋值
                assert len(index) == len(
                    path_average_speeds), "The number of path average speeds must match the number of indices"

                # 遍历索引和路径平均速度张量，将它们赋值到 average_speeds
                for idx, speed_tensor in zip(index, path_average_speeds):
                    average_speeds[idx] = speed_tensor
    return all_unique_path_ids, average_speeds, records
