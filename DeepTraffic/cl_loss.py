import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from SupConLoss import SupConLoss
# from sklearn.cluster import KMeans
from kmeans_pytorch import kmeans




def get_traj_cl_loss(gps_traj_rep, route_traj_rep, tau=0.07):
    gps_traj_rep = F.normalize(gps_traj_rep, dim=1)
    route_traj_rep = F.normalize(route_traj_rep, dim=1)
    logits = torch.mm(gps_traj_rep, route_traj_rep.t()) / tau
    gps_sim = torch.mm(gps_traj_rep, route_traj_rep.t())
    route_sim = torch.mm(route_traj_rep, gps_traj_rep.t())
    targets = F.softmax((gps_sim + route_sim) / 2 * tau, dim=-1)
    ce = nn.CrossEntropyLoss(reduction='none')
    gps_loss = ce(logits, targets)
    route_loss = ce(logits.T, targets.T)
    traj_cl_loss = (gps_loss + route_loss) / 2.0
    traj_cl_loss = traj_cl_loss.mean()
    return traj_cl_loss

def get_road_cl_loss(gps_road_rep, route_road_rep, y_label, tau=0.07):
    criterion = SupConLoss(temperature=tau)
    gps_road_rep = F.normalize(gps_road_rep, dim=1)
    route_road_rep = F.normalize(route_road_rep, dim=1)
    features = torch.cat([gps_road_rep.unsqueeze(1), route_road_rep.unsqueeze(1)], dim=1)  # bs*views*f_dim
    road_cl_loss = criterion(features, y_label)
    return road_cl_loss

def get_traj_cluster_loss(gps_traj_rep, route_traj_rep, device, tau=0.07, n_clusters=64):
    # device = torch.device('cuda:0')
    data = torch.cat([gps_traj_rep, route_traj_rep], dim=0)
    cluster_ids_x, cluster_centers = kmeans(
        X=data, num_clusters=n_clusters, distance='euclidean', device=device
    )

    cluster_centers = cluster_centers.cuda()
    # 计算 GPS 轨迹和聚类中心之间的相似度矩阵
    gps_cluster_sim = torch.mm(gps_traj_rep, cluster_centers.t())
    # 计算路线规划和聚类中心之间的相似度矩阵
    route_cluster_sim = torch.mm(route_traj_rep, cluster_centers.t())

    targets = F.softmax((gps_cluster_sim + route_cluster_sim) / 2 * tau, dim=-1)

    gps_cluster_loss = nn.CrossEntropyLoss(reduction='none')(gps_cluster_sim, targets)  # 计算 GPS 轨迹和聚类中心之间的交叉熵损失
    route_cluster_loss = nn.CrossEntropyLoss(reduction='none')(route_cluster_sim, targets)  # 计算路线规划和聚类中心之间的交叉熵损失

    traj_cluster_loss = (gps_cluster_loss + route_cluster_loss) / 2.0
    traj_cluster_loss = traj_cluster_loss.mean()
    return traj_cluster_loss

def get_road_cluster_loss(gps_road_rep, route_road_rep, tau=0.07, n_clusters=64, dev_id=0):
    data = torch.cat([gps_road_rep, route_road_rep], dim=0)
    cluster_ids_x, cluster_centers = kmeans(
        X=data, num_clusters=n_clusters, distance='euclidean', device=torch.device('cuda:{}'.format(dev_id)), tqdm_flag=False
    )

    cluster_centers = cluster_centers.cuda()
    # 计算 GPS 轨迹和聚类中心之间的相似度矩阵
    gps_cluster_sim = torch.mm(gps_road_rep, cluster_centers.t())
    # 计算路线规划和聚类中心之间的相似度矩阵
    route_cluster_sim = torch.mm(route_road_rep, cluster_centers.t())

    targets = F.softmax((gps_cluster_sim + route_cluster_sim) / 2 * tau, dim=-1)

    gps_cluster_loss = nn.CrossEntropyLoss(reduction='none')(gps_cluster_sim, targets)  # 计算 GPS 轨迹和聚类中心之间的交叉熵损失
    route_cluster_loss = nn.CrossEntropyLoss(reduction='none')(route_cluster_sim, targets)  # 计算路线规划和聚类中心之间的交叉熵损失

    road_cluster_loss = (gps_cluster_loss + route_cluster_loss) / 2.0
    road_cluster_loss = road_cluster_loss.mean()
    return road_cluster_loss


def get_traj_match_loss(gps_traj_rep, route_traj_rep, model, batch_size=64, tau=0.07):
    batch_size = gps_traj_rep.shape[0]
    gps_traj_rep = F.normalize(gps_traj_rep, dim=1)
    route_traj_rep = F.normalize(route_traj_rep, dim=1)  # 列（特征维度）上的相对比例被保留下来，按行标准化，使得每行（64个样本中的每一个）的模长变为1

    # gps_traj_all = torch.cat([gps_traj_rep.t(), model.gps_queue.clone().detach()], dim=1) # 256 x 2048+64
    # route_traj_all = torch.cat([route_traj_rep.t(), model.route_queue.clone().detach()], dim=1)
    # 以下计算不同轨迹之间的相似度
    sim_g2r = gps_traj_rep @ route_traj_rep.t() / tau
    sim_r2g = route_traj_rep @ gps_traj_rep.t() / tau

    weight_g2r = F.softmax(sim_g2r, dim=1)
    weight_r2g = F.softmax(sim_r2g, dim=1)

    sim_g2r.fill_diagonal_(0)
    sim_r2g.fill_diagonal_(0)

    # select a negative route for each gps
    route_traj_rep_neg = []
    for i in range(batch_size):
        neg_idx = torch.multinomial(weight_g2r[i], 1).item()
        route_traj_rep_neg.append(route_traj_rep[neg_idx])
    route_traj_rep_neg = torch.stack(route_traj_rep_neg, dim=0)

    # select a negative gps for each route
    gps_traj_rep_neg = []
    for i in range(batch_size):
        neg_idx = torch.multinomial(weight_r2g[i], 1).item()
        gps_traj_rep_neg.append(gps_traj_rep[neg_idx])
    gps_traj_rep_neg = torch.stack(gps_traj_rep_neg, dim=0)


    # 每一个GR pair都有两个负样本 GR’ 和 G‘R，flat之后分为3组，GR,GR',G'R 64*3 x 256

    pos_pair = torch.cat([route_traj_rep, gps_traj_rep], dim=1)
    neg_pair1 = torch.cat([route_traj_rep_neg, gps_traj_rep], dim=1)
    neg_pair2 = torch.cat([route_traj_rep, gps_traj_rep_neg], dim=1)

    all_pair = torch.cat([pos_pair, neg_pair1, neg_pair2], dim=0)

    pred = model.matching_predictor(all_pair)  # 第一列概率对应于"不匹配"类别（标签0），第二列对应于"匹配"类别（标签1）

    label = torch.cat([torch.ones(batch_size, dtype=torch.long), torch.zeros(2 * batch_size, dtype=torch.long)],dim=0).cuda() # 1 x 64*3
    loss = F.cross_entropy(pred, label)
    return loss


def get_road_match_loss(speed_rep, time_rep, unique_path_ids, truth_interval, interval_emb, model, batch_size=64):
    gps_time_tensor = torch.tensor(speed_rep, dtype=torch.float32)
    truth_interval_tensor = torch.tensor(truth_interval, dtype=torch.float32)
    gps_time_tensor = gps_time_tensor.to('cuda:0')
    truth_interval_tensor = truth_interval_tensor.to('cuda:0')
    # 3. 使用interval_emb函数将理论行程时间转换为特征向量
    gps_transformed_time = interval_emb(gps_time_tensor)
    time_rep = time_rep.to('cuda:0')
    route_time_value = interval_emb.decode(time_rep)
    # 逐元素相加并取平均值
    zero_rows = torch.all(time_rep == 0, dim=1)
    # 使用 where 函数简化代码
    fused_tensor = torch.where(zero_rows.unsqueeze(1), gps_transformed_time, (time_rep + gps_transformed_time) / 2)
    fused_values = interval_emb.decode(fused_tensor)
    trans_truth_interval = interval_emb(truth_interval_tensor)
    try:
        assert time_rep.shape == gps_transformed_time.shape
    except AssertionError as e:
        print(f"断言失败：{e}")

    # 4. 计算两个特征向量之间的相似度损失
    # 这里假设 `interval_emb` 输出的特征向量形状与 `time_rep` 相同
    # 使用余弦相似度来衡量两个向量之间的相似度
    similarity = F.cosine_similarity(time_rep, gps_transformed_time)
    similarity_truth =  F.cosine_similarity(fused_tensor, trans_truth_interval)

    # 5. 构建损失函数，可以使用1 - cosine_similarity作为损失
    # 因为cosine_similarity范围是[-1, 1]，我们希望它尽可能接近1，即相似度高
    loss = 1 - similarity.mean()  # 平均所有样本的相似度损失
    loss_truth = 1 - similarity_truth.mean()

    return loss, loss_truth, fused_values, truth_interval_tensor
