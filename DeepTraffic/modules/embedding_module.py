import itertools

import torch
from torch import nn
import numpy as np
import math
import torch.nn.functional as F
from model.temporal_attention import TemporalAttentionLayer


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


class LearnableIntervalEmbedding(nn.Module):
    def __init__(self, num_bins, hidden_size, min_time=0, max_time=100):
        super(LearnableIntervalEmbedding, self).__init__()
        self.num_bins = num_bins
        self.hidden_size = hidden_size
        # 初始化区间的边界为均匀分布
        self.bin_boundaries = nn.Parameter(torch.linspace(min_time, max_time, steps=num_bins + 1))
        self.emb = nn.Embedding(num_bins, hidden_size)
        self.bin_centers = (self.bin_boundaries[:-1] + self.bin_boundaries[1:]) / 2

    def forward(self, x):
        # 找到每个输入值所属的区间索引
        bin_indices = torch.bucketize(x, self.bin_boundaries) - 1
        # 确保索引在有效范围内
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1).to('cuda:0')
        # 获取对应的嵌入向量
        output = self.emb(bin_indices)
        return output

    def decode(self, embeddings):
        # 计算输入嵌入向量与所有分箱嵌入之间的余弦相似度
        # embeddings: [batch_size, hidden_size]
        # self.emb.weight: [num_bins, hidden_size]
        similarities = F.cosine_similarity(embeddings.unsqueeze(1), self.emb.weight, dim=-1)
        # 找到最相似的分箱索引
        closest_bin_indices = torch.argmax(similarities, dim=1)
        closest_bin_indices = closest_bin_indices.to('cuda:0')
        self.bin_centers = self.bin_centers.to('cuda:0')
        # 使用找到的分箱索引查找对应的区间中心点
        decoded_value = self.bin_centers[closest_bin_indices]
        return decoded_value  # 确保输出形状与输入一致


class EmbeddingModule(nn.Module):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               dropout):
    super(EmbeddingModule, self).__init__()
    self.node_features = node_features
    self.edge_features = edge_features
    # self.memory = memory
    self.delta_embedding = LearnableIntervalEmbedding(300, embedding_dimension)
    self.route_padding_vec = torch.zeros((1, embedding_dimension))
    self.neighbor_finder = neighbor_finder
    self.time_encoder = time_encoder
    self.n_layers = n_layers
    self.n_node_features = n_node_features
    self.n_edge_features = n_edge_features
    self.n_time_features = n_time_features
    self.dropout = dropout
    self.embedding_dimension = embedding_dimension
    self.device = device
    # 创建 IntervalEmbedding 实例
    self.interval_emb = LearnableIntervalEmbedding(num_bins=300, hidden_size=128)

  def compute_embedding(self, memory, source_nodes, timestamps, route_len, intervals, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    return NotImplemented


class IdentityEmbedding(EmbeddingModule):
  def compute_embedding(self, memory, source_nodes, timestamps, route_len, intervals, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    return memory[source_nodes, :]


class TimeEmbedding(EmbeddingModule):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True, n_neighbors=1):
    super(TimeEmbedding, self).__init__(node_features, edge_features, memory,
                                        neighbor_finder, time_encoder, n_layers,
                                        n_node_features, n_edge_features, n_time_features,
                                        embedding_dimension, device, dropout)

    class NormalLinear(nn.Linear):
      # From Jodie code
      def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
          self.bias.data.normal_(0, stdv)

    self.embedding_layer = NormalLinear(1, self.n_node_features)

  def compute_embedding(self, memory, source_nodes, timestamps, route_len, intervals, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    source_embeddings = memory[source_nodes, :] * (1 + self.embedding_layer(time_diffs.unsqueeze(1)))

    return source_embeddings


class GraphEmbedding(EmbeddingModule):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True):
    super(GraphEmbedding, self).__init__(node_features, edge_features, memory,
                                         neighbor_finder, time_encoder, n_layers,
                                         n_node_features, n_edge_features, n_time_features,
                                         embedding_dimension, device, dropout)

    self.use_memory = use_memory
    # 创建 IntervalEmbedding 实例
    self.interval_emb = LearnableIntervalEmbedding(num_bins=300, hidden_size=128)
    self.device = device

  def compute_embedding(self, memory, source_nodes, timestamps, route_len, intervals, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    """Recursive implementation of curr_layers temporal graph attention layers.

    src_idx_l [batch_size]: users / items input ids.
    cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
    curr_layers [scalar]: number of temporal convolutional layers to stack.
    num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
    """

    assert (n_layers >= 0)

    source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
    timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

    # query node always has the start time -> time span == 0
    # source_nodes_time_embedding = self.time_encoder(torch.zeros_like(
    #   timestamps_torch))
    source_nodes_time_embedding = self.time_encoder(timestamps_torch)
    source_node_features = self.node_features[source_nodes_torch, :]
    # 假设 intervals 是多余包含一个时间间隔的列表，这里将其转换为 PyTorch Tensor
    intervals_tensor = torch.tensor(intervals, dtype=torch.float32)
    # 获取时间间隔的嵌入表示
    source_node_delta_embedding = self.interval_emb(intervals_tensor)

    if source_node_delta_embedding.shape[0] != (len(source_nodes)):
        # 初始化src_node_delta_embedding和dest_node_delta_embedding
        src_node_delta_embedding = []
        dest_node_delta_embedding = []

        # 跟踪上一个序列的结束位置
        start_index = 0

        for i, seq_len in enumerate(route_len):
            # 计算当前序列的结束位置
            end_index = start_index + seq_len

            if seq_len > 1:
                # 构建src_node_delta_embedding，去掉end_index对应的元素
                src_node_delta_embedding.append(source_node_delta_embedding[start_index:end_index - 1])

                # 构建dest_node_delta_embedding，去掉start_index对应的元素
                # 并将剩余的元素左移一位
                # 使用torch.roll进行左移操作，shifts参数为-1表示左移一位
                dest_node_delta_embedding.append(
                    torch.roll(source_node_delta_embedding[start_index:end_index], shifts=-1, dims=0)[:-1])

            # 更新下一个序列的开始位置
            start_index = end_index

        # 将列表中的张量拼接成一个大的张量
        src_node_delta_embedding = torch.cat(src_node_delta_embedding, dim=0)
        dest_node_delta_embedding = torch.cat(dest_node_delta_embedding, dim=0)
        concatenated_delta_embeddings = torch.cat((src_node_delta_embedding,
                                                   dest_node_delta_embedding,), dim=0).to(self.device)
    else:
        concatenated_delta_embeddings = source_node_delta_embedding.to(self.device)

    if self.use_memory:
      source_node_features = memory[source_nodes, :] + torch.cat((source_node_features, concatenated_delta_embeddings), dim=1)# 引入了记忆模块来更新节点特征，还应加上路段经过时间特征

    if n_layers == 0:
      return source_node_features
    else:

      source_node_conv_embeddings = self.compute_embedding(memory,
                                                           source_nodes,
                                                           timestamps, route_len, intervals,
                                                           n_layers=n_layers - 1,
                                                           n_neighbors=n_neighbors)

      neighbors, edge_idxs, edge_times, nei_intervals = self.neighbor_finder.get_temporal_neighbor(
        source_nodes,
        timestamps,
        n_neighbors=n_neighbors)

      neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)

      edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)

      edge_deltas = timestamps[:, np.newaxis] - edge_times  # 可以同时计算与历史同时段的差值（但考虑周末）

      edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

      neighbors = neighbors.flatten()
      nei_interval = list(itertools.chain.from_iterable(nei_intervals))
      neighbor_embeddings = self.compute_embedding(memory,
                                                   neighbors,
                                                   np.repeat(timestamps, n_neighbors),
                                                   None, nei_interval,
                                                   n_layers=n_layers - 1,
                                                   n_neighbors=n_neighbors)

      effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
      neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)
      edge_time_embeddings = self.time_encoder(edge_deltas_torch)

      edge_features = self.edge_features[edge_idxs, :]

      mask = neighbors_torch == -1

      source_embedding = self.aggregate(n_layers, source_node_conv_embeddings,
                                        source_nodes_time_embedding,
                                        neighbor_embeddings,
                                        edge_time_embeddings,
                                        edge_features,
                                        mask)

      return source_embedding

  def aggregate(self, n_layers, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    return NotImplemented


class GraphSumEmbedding(GraphEmbedding):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True):
    super(GraphSumEmbedding, self).__init__(node_features=node_features,
                                            edge_features=edge_features,
                                            memory=memory,
                                            neighbor_finder=neighbor_finder,
                                            time_encoder=time_encoder, n_layers=n_layers,
                                            n_node_features=n_node_features,
                                            n_edge_features=n_edge_features,
                                            n_time_features=n_time_features,
                                            embedding_dimension=embedding_dimension,
                                            device=device,
                                            n_heads=n_heads, dropout=dropout,
                                            use_memory=use_memory)
    self.linear_1 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension + n_time_features +
                                                         n_edge_features, embedding_dimension)
                                         for _ in range(n_layers)])
    self.linear_2 = torch.nn.ModuleList(
      [torch.nn.Linear(embedding_dimension + n_node_features + n_time_features,
                       embedding_dimension) for _ in range(n_layers)])

  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    neighbors_features = torch.cat([neighbor_embeddings, edge_time_embeddings, edge_features],
                                   dim=2)
    neighbor_embeddings = self.linear_1[n_layer - 1](neighbors_features)
    neighbors_sum = torch.nn.functional.relu(torch.sum(neighbor_embeddings, dim=1))

    source_features = torch.cat([source_node_features,
                                 source_nodes_time_embedding.squeeze()], dim=1)
    source_embedding = torch.cat([neighbors_sum, source_features], dim=1)
    source_embedding = self.linear_2[n_layer - 1](source_embedding)

    return source_embedding


class GraphAttentionEmbedding(GraphEmbedding):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True):
    super(GraphAttentionEmbedding, self).__init__(node_features, edge_features, memory,
                                                  neighbor_finder, time_encoder, n_layers,
                                                  n_node_features, n_edge_features,
                                                  n_time_features,
                                                  embedding_dimension, device,
                                                  n_heads, dropout,
                                                  use_memory)

    self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(
      n_node_features=n_node_features,
      n_neighbors_features=n_node_features,
      n_edge_features=n_edge_features,
      time_dim=n_time_features,
      n_head=n_heads,
      dropout=dropout,
      output_dimension=n_node_features)
      for _ in range(n_layers)])

  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    attention_model = self.attention_models[n_layer - 1]

    source_embedding, _ = attention_model(source_node_features,
                                          source_nodes_time_embedding,
                                          neighbor_embeddings,
                                          edge_time_embeddings,
                                          edge_features,
                                          mask)

    return source_embedding


def get_embedding_module(module_type, node_features, edge_features, memory, neighbor_finder,
                         time_encoder, n_layers, n_node_features, n_edge_features, n_time_features,
                         embedding_dimension, device,
                         n_heads=2, dropout=0.1, n_neighbors=None,
                         use_memory=True):
  if module_type == "graph_attention":
    return GraphAttentionEmbedding(node_features=node_features,
                                    edge_features=edge_features,
                                    memory=memory,
                                    neighbor_finder=neighbor_finder,
                                    time_encoder=time_encoder,
                                    n_layers=n_layers,
                                    n_node_features=n_node_features,
                                    n_edge_features=n_edge_features,
                                    n_time_features=n_time_features,
                                    embedding_dimension=embedding_dimension,
                                    device=device,
                                    n_heads=n_heads, dropout=dropout, use_memory=use_memory)
  elif module_type == "graph_sum":
    return GraphSumEmbedding(node_features=node_features,
                              edge_features=edge_features,
                              memory=memory,
                              neighbor_finder=neighbor_finder,
                              time_encoder=time_encoder,
                              n_layers=n_layers,
                              n_node_features=n_node_features,
                              n_edge_features=n_edge_features,
                              n_time_features=n_time_features,
                              embedding_dimension=embedding_dimension,
                              device=device,
                              n_heads=n_heads, dropout=dropout, use_memory=use_memory)

  elif module_type == "identity":
    return IdentityEmbedding(node_features=node_features,
                             edge_features=edge_features,
                             memory=memory,
                             neighbor_finder=neighbor_finder,
                             time_encoder=time_encoder,
                             n_layers=n_layers,
                             n_node_features=n_node_features,
                             n_edge_features=n_edge_features,
                             n_time_features=n_time_features,
                             embedding_dimension=embedding_dimension,
                             device=device,
                             dropout=dropout)
  elif module_type == "time":
    return TimeEmbedding(node_features=node_features,
                         edge_features=edge_features,
                         memory=memory,
                         neighbor_finder=neighbor_finder,
                         time_encoder=time_encoder,
                         n_layers=n_layers,
                         n_node_features=n_node_features,
                         n_edge_features=n_edge_features,
                         n_time_features=n_time_features,
                         embedding_dimension=embedding_dimension,
                         device=device,
                         dropout=dropout,
                         n_neighbors=n_neighbors)
  else:
    raise ValueError("Embedding Module {} not supported".format(module_type))


