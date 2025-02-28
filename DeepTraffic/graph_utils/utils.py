import itertools

import numpy as np
import torch
from datetime import datetime


def get_middle_timestamps_datetime(timestamps):
  middle_timestamps = []
  deltas = []
  for i in range(len(timestamps) - 1):
    delta = timestamps[i + 1] - timestamps[i]
    middle_time = timestamps[i] + (delta / 2)
    middle_timestamps.append(middle_time)
    deltas.append(delta)
  return middle_timestamps, deltas


def convert_timestamps_to_minutes(timestamp_list):
  minute_list = []
  for timestamp in timestamp_list:
    if timestamp == 0:
      minute_list.append(0)
      continue
    dt = datetime.fromtimestamp(timestamp)
    minute_of_day = dt.hour * 60 + dt.minute
    minute_list.append(minute_of_day)
  return minute_list


class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h = self.act(self.fc1(x))
    return self.fc2(h)


class MLP(torch.nn.Module):
  def __init__(self, dim, drop=0.3):
    super().__init__()
    self.fc_1 = torch.nn.Linear(dim, 80)
    self.fc_2 = torch.nn.Linear(80, 10)
    self.fc_3 = torch.nn.Linear(10, 1)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1)


class EarlyStopMonitor(object):
  def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
    self.max_round = max_round
    self.num_round = 0

    self.epoch_count = 0
    self.best_epoch = 0

    self.last_best = None
    self.higher_better = higher_better
    self.tolerance = tolerance

  def early_stop_check(self, curr_val):
    if not self.higher_better:
      curr_val *= -1
    if self.last_best is None:
      self.last_best = curr_val
    elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
      self.last_best = curr_val
      self.num_round = 0
      self.best_epoch = self.epoch_count
    else:
      self.num_round += 1

    self.epoch_count += 1

    return self.num_round >= self.max_round


class RandEdgeSampler(object):
  def __init__(self, src_list, dst_list, seed=None):
    self.seed = None
    self.src_list = np.unique(src_list)
    self.dst_list = np.unique(dst_list)

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def sample(self, size):
    if self.seed is None:
      src_index = np.random.randint(0, len(self.src_list), size)
      dst_index = np.random.randint(0, len(self.dst_list), size)
    else:

      src_index = self.random_state.randint(0, len(self.src_list), size)
      dst_index = self.random_state.randint(0, len(self.dst_list), size)
    return self.src_list[src_index], self.dst_list[dst_index]

  def reset_random_state(self):
    self.random_state = np.random.RandomState(self.seed)


def get_neighbor_finder(data, road_timestamps, uniform, max_node_idx=None):
  max_node_idx = max(data.sources.max(), data.destinations.max())if max_node_idx is None else max_node_idx
  # 应用 get_middle_timestamps_datetime 函数到 road_timestamps_batch 的每一个内部列表
  road_midl_stamps_batch = [get_middle_timestamps_datetime(ts)[0] for ts in road_timestamps]
  intervals = [get_middle_timestamps_datetime(ts)[1] for ts in road_timestamps]

  src_timestamps_batch = [timestamps[:-1] for timestamps in road_midl_stamps_batch]
  src_timestamps = list(itertools.chain.from_iterable(src_timestamps_batch))
  des_timestamps_batch = [timestamps[1:] for timestamps in road_midl_stamps_batch]
  des_timestamps = list(itertools.chain.from_iterable(des_timestamps_batch))
  src_intervals = [interval[:-1] for interval in intervals]
  src_interval1 = list(itertools.chain.from_iterable(src_intervals))
  des_intervals = [interval[1:] for interval in intervals]
  des_interval1 = list(itertools.chain.from_iterable(des_intervals))
  adj_list = [[] for _ in range(max_node_idx + 1)]
  for source, destination, edge_idx, src_timestamp, des_timestamp, src_interval,des_interval in zip(data.sources, data.destinations,data.edge_idxs,
                                                                           src_timestamps, des_timestamps,src_interval1,des_interval1
                                                                           ):
    adj_list[source].append((destination, edge_idx, src_timestamp,src_interval))  # 已知所有的各时间的交互，直接得到带时间的邻居
    adj_list[destination].append((source, edge_idx, des_timestamp, des_interval))

  return NeighborFinder(adj_list, uniform=uniform)


class NeighborFinder:
  def __init__(self, adj_list, uniform=False, seed=None):
    self.node_to_neighbors = []
    self.node_to_edge_idxs = []
    self.node_to_edge_timestamps = []
    self.node_to_edge_intervals = []

    for neighbors in adj_list:
      # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
      # We sort the list based on timestamp
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
      self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
      self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
      self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))
      self.node_to_edge_intervals.append(np.array([x[3] for x in sorted_neighhbors]))

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_before(self, src_idx, cut_time):
    """
    Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

    Returns 3 lists: neighbors, edge_idxs, timestamps

    """
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

    return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i], self.node_to_edge_intervals[src_idx][:i]

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    assert (len(source_nodes) == len(timestamps))

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    # NB! All interactions described in these matrices are sorted in each row by time
    neighbors = np.full((len(source_nodes), tmp_n_neighbors), -1, dtype=np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    intervals = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.float32)

    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      source_neighbors, source_edge_idxs, source_edge_times, source_intervals = self.find_before(source_node,
                                                   timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

      if len(source_neighbors) > 0 and n_neighbors > 0:
        if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
          # sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)
          #
          # neighbors[i, :] = source_neighbors[sampled_idx]
          # edge_times[i, :] = source_edge_times[sampled_idx]
          # edge_idxs[i, :] = source_edge_idxs[sampled_idx]
          #
          # # re-sort based on time
          # pos = edge_times[i, :].argsort()
          # neighbors[i, :] = neighbors[i, :][pos]
          # edge_times[i, :] = edge_times[i, :][pos]
          # edge_idxs[i, :] = edge_idxs[i, :][pos]
          target_time_in_minutes = convert_timestamps_to_minutes([timestamp])[0]
          edge_times_in_minutes = convert_timestamps_to_minutes(source_edge_times)
          time_diffs = np.abs(np.array(edge_times_in_minutes) - target_time_in_minutes)

          closest_indices = np.argsort(time_diffs)[:n_neighbors]

          # 如果找到的最近邻少于 n_neighbors，计算还缺少多少个
          missing_neighbors = n_neighbors - len(closest_indices)

          # 如果需要添加额外的邻居
          if missing_neighbors > 0:
            if len(time_diffs) > 1:
              # 对 time_diffs 进行排序并获取次最小值的索引
              sorted_indices = np.argsort(time_diffs)
              min_time_diff_index = sorted_indices[0]
            else:
              min_time_diff_index = np.argmin(time_diffs)
            # 重复添加缺失数量的最小值索引
            additional_indices = np.full(missing_neighbors, min_time_diff_index, dtype=int)
            closest_indices = np.concatenate((closest_indices, additional_indices))

          # 从最近邻中采样
          sampled_source_neighbors = [source_neighbors[idx] for idx in closest_indices[:n_neighbors]]
          sampled_source_edge_times = [source_edge_times[idx] for idx in closest_indices[:n_neighbors]]
          sampled_source_edge_idxs = [source_edge_idxs[idx] for idx in closest_indices[:n_neighbors]]
          sampled_source_intervals = [source_intervals[idx] for idx in closest_indices[:n_neighbors]]

          # 将结果存储到 neighbors, edge_times, 和 edge_idxs 中
          neighbors[i, :] = sampled_source_neighbors
          edge_times[i, :] = sampled_source_edge_times
          edge_idxs[i, :] = sampled_source_edge_idxs
          intervals[i, :] = sampled_source_intervals

          # 重新排序以按时间顺序排列
          pos = edge_times[i, :].argsort()
          neighbors[i, :] = neighbors[i, :][pos]
          edge_times[i, :] = edge_times[i, :][pos]
          edge_idxs[i, :] = edge_idxs[i, :][pos]
          intervals[i, :] = intervals[i, :][pos]
        else:
          # Take most recent interactions
          source_edge_times = source_edge_times[-n_neighbors:]
          source_neighbors = source_neighbors[-n_neighbors:]
          source_edge_idxs = source_edge_idxs[-n_neighbors:]
          source_intervals = source_intervals[-n_neighbors:]

          assert (len(source_neighbors) <= n_neighbors)
          assert (len(source_edge_times) <= n_neighbors)
          assert (len(source_edge_idxs) <= n_neighbors)
          assert (len(source_intervals) <= n_neighbors)

          neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
          edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
          edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs
          intervals[i, n_neighbors - len(source_intervals):] = source_intervals

    return neighbors, edge_idxs, edge_times, intervals
