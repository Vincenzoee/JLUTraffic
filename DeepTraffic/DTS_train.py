import itertools
import pickle
import pandas
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup, AdamW
import wandb
import pandas as pd
from evaluation.graph_evaluation.evaluation import eval_edge_prediction
from gps_utils.tools import visual
from graph_utils.data_processing import compute_time_statistics
from graph_utils.utils import get_neighbor_finder, RandEdgeSampler, EarlyStopMonitor
from model import tgn
from model.tgn import TGN
from utils import weight_init
from dataloader import get_train_loader, random_mask, NodeEmbeddingModel, GraphData, TGNConfig, get_loader, \
    get_middle_timestamps_datetime, GPSConfig
from utils import setup_seed
import numpy as np
import json
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from DTS import DTSModel
from cl_loss import get_traj_cl_loss, get_road_cl_loss, get_traj_cluster_loss, get_traj_match_loss, get_road_match_loss
from dcl import DCL
import os

dev_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(dev_id)
torch.cuda.set_device(dev_id)
torch.set_num_threads(10)


def compute_losses(fused_values, truth_interval):
    # 实例化损失函数
    mse_loss_fn = nn.MSELoss(reduction='mean')
    # 计算均方误差 (MSE)
    mse = mse_loss_fn(fused_values, truth_interval)
    # 计算均方根误差 (RMSE)
    rmse = torch.sqrt(mse)
    # 计算平均绝对误差 (MAE)
    mae = torch.abs(fused_values - truth_interval).mean()
    # Avoid division by zero by adding a small epsilon to actual values
    epsilon = 2
    actual = torch.where(truth_interval == 0, epsilon, truth_interval)
    # Calculate MAPE
    mape = torch.mean(torch.abs((actual - fused_values) / actual)) * 100
    # 返回包含所有损失的字典
    return mse.item(), rmse.item(), mae.item(), mape.item()


def vali(model, vali_loader, val_rand_sampler, n_neighbors, use_memory, g_criterion, log_path, epoch, test=False):
    per_epoch_train_g_loss = []
    per_epoch_match_loss = []
    per_epoch_ultimate_loss = []
    per_epoch_mse = []
    per_epoch_mae = []
    per_epoch_rmse = []
    per_epoch_mape = []

    epo = epoch
    model = model.eval()
    print('HERE in vali!!!!!!!!!!!!!!!' if not test else 'This is a test')
    with torch.no_grad():
        val_ap, val_auc = eval_edge_prediction(model=model.tgn,
                                               negative_edge_sampler=val_rand_sampler,
                                               data=vali_loader,
                                               n_neighbors=n_neighbors
                                               )
        if use_memory:
            val_memory_backup = model.tgn.memory.backup_memory()
            # Restore memory we had at the end of training to be used when validating on new nodes.
            # Also backup memory after validation so it can be used for testing (since test edges are
            # strictly later in time than validation edges)
            model.tgn.memory.restore_memory(val_memory_backup)

        for idx, batch in enumerate(vali_loader):
            x_indo_features_batch = batch['x_indo_features'],
            x_exo_features_batch = batch['x_exo_features'],
            x_opath_batch = batch['x_opath'],
            x_time_feature_batch = batch['x_time_feature'],
            y_indo_features_batch = batch['y_indo_features'],
            y_opath_batch = batch['y_opath'],
            y_time_feature_batch = batch['y_time_feature'],
            outputs, seq_endogenous_y = model.encode_gps(x_indo_features_batch, x_exo_features_batch, x_opath_batch,
                                                         x_time_feature_batch, y_indo_features_batch, y_opath_batch,
                                                         y_time_feature_batch, gps_length=None)
            y_valid = torch.logical_not(torch.eq(y_opath_batch[0], -1.0))
            y_valid = y_valid.to('cuda:0')
            loss_per_point = g_criterion(outputs, seq_endogenous_y)
            # 应用掩码，忽略填充值的影响
            masked_loss_per_point = loss_per_point.squeeze() * y_valid
            # 计算总的损失，除以有效元素的数量来获得平均损失
            batch_gps_loss = masked_loss_per_point.sum() / y_valid.sum()
            per_epoch_train_g_loss.append(batch_gps_loss.item())

            times, unique_path_ids, truth_interval = model.calculate_average_speeds(x_opath_batch, y_opath_batch, x_indo_features_batch,
                                                                                    x_time_feature_batch, y_time_feature_batch, outputs)
            average_interval_route = model.tgn.memory.get_memory(unique_path_ids)[:, -128:]
            batch_match_loss, batch_truth_loss, fused_values, truth_interval_tensor\
                = get_road_match_loss(times, average_interval_route, unique_path_ids, truth_interval,
                                      model.tgn.embedding_module.interval_emb, model)
            mse, rmse, mae, mape = compute_losses(fused_values, truth_interval_tensor)
            per_epoch_mse.append(mse)
            per_epoch_rmse.append(rmse)
            per_epoch_mae.append(mae)
            per_epoch_mape.append(mape)
            per_epoch_match_loss.append(batch_match_loss.item())
            per_epoch_ultimate_loss.append(batch_truth_loss.item())

            if idx % 30 == 0:
                input = x_indo_features_batch[0].detach().cpu().numpy()
                true = y_indo_features_batch[0].detach().cpu().numpy()
                pred = outputs.detach().cpu().numpy()
                gt = np.concatenate((input[0+epo, -50:, -1], true[0+epo, :50, -1]), axis=0)
                pd = np.concatenate((input[0+epo, -50:, -1], pred[0+epo, :50, -1]), axis=0)
                if test:
                    visual(gt, pd, os.path.join(log_path, 'test_' + str(idx) + '.pdf'))
                else:
                    visual(gt, pd, os.path.join(log_path, 'vali_' + str(epo) + '_' + str(idx) + '.pdf'))
        aver_epoch_train_g_loss = np.average(per_epoch_train_g_loss)
        aver_epoch_match_loss = np.average(per_epoch_match_loss)
        aver_epoch_ultimate_loss = np.average(per_epoch_ultimate_loss)
        aver_epoch_mse = np.mean(per_epoch_mse)
        aver_epoch_rmse = np.mean(per_epoch_rmse)
        aver_epoch_mae = np.mean(per_epoch_mae)
        aver_epoch_mape = np.mean(per_epoch_mape)

        model.train()
        return aver_epoch_train_g_loss, aver_epoch_match_loss, aver_epoch_ultimate_loss, val_ap, val_auc, \
            aver_epoch_mse, aver_epoch_rmse, aver_epoch_mae, aver_epoch_mape


def train(config):

    city = config['city']

    vocab_size = config['vocab_size']
    num_samples = config['num_samples']
    data_path = config['data_path']
    adj_path = config['adj_path']
    retrain = config['retrain']
    save_path = config['save_path']

    num_worker = config['num_worker']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    warmup_step = config['warmup_step']
    weight_decay = config['weight_decay']

    route_min_len = config['route_min_len']
    route_max_len = config['route_max_len']
    gps_min_len = config['gps_min_len']
    gps_max_len = config['gps_max_len']

    road_feat_num = config['road_feat_num']
    road_embed_size = config['road_embed_size']
    gps_feat_num = config['gps_feat_num']
    gps_embed_size = config['gps_embed_size']
    route_embed_size = config['route_embed_size']

    hidden_size = config['hidden_size']
    drop_route_rate = config['drop_route_rate']  # route_encoder
    drop_edge_rate = config['drop_edge_rate']   # gat
    drop_road_rate = config['drop_road_rate']   # sharedtransformer

    verbose = config['verbose']
    version = config['version']
    seed = config['random_seed']

    mask_length = config['mask_length']
    mask_prob = config['mask_prob']

    # 其余的配置项以同样的方式从config字典中读取
    use_memory = config['use_memory']
    embedding_module = config['embedding_module']
    message_function = config['message_function']
    memory_updater = config['memory_updater']
    aggregator = config['aggregator']
    memory_update_at_start = config['memory_update_at_start']
    message_dim = config['message_dim']
    memory_dim = config['memory_dim']
    different_new_nodes = config['different_new_nodes']
    uniform = config['uniform']
    randomize_features = config['randomize_features']
    use_destination_embedding_in_message = config['use_destination_embedding_in_message']
    use_source_embedding_in_message = config['use_source_embedding_in_message']
    dyrep = config['dyrep']
    n_degree = config['n_degree']  # 直接访问，没有默认值
    n_head = config['n_head']  # 直接访问，没有默认值
    n_layer = config['n_layer']  # 直接访问，没有默认值

    # 设置随机种子
    setup_seed(seed)

    csv_file_path = './data/edge_features.csv'
    # 使用 pandas 读取 CSV 文件
    edge_features_df = pandas.read_csv(csv_file_path)
    edge_index = np.load(adj_path)
    init_road_emb = torch.load(r'.\data\init_w2v_road_emb.pt', map_location='cuda:{}'.format(dev_id))
    train_loader,val_loader,test_loader,vocab_size,node_features,full_ngh_finder,train_rand_sampler,train_ngh_finder,test_rand_sampler,val_rand_sampler,device,\
        = get_loader(data_path, edge_index, edge_features_df, batch_size, num_worker, route_min_len, route_max_len, gps_min_len, gps_max_len,
                     num_samples, seed, vocab_size, road_embed_size, init_road_emb, city)
    print('dataset is ready.')
    vocab_size = vocab_size
    edge_features = np.load('./data/combined_temp_edge_features.npy')
    with open(r'.\data\1w_nodes_dict.pkl', 'rb') as f:
        nodes_dict = pickle.load(f)

    tgn_config = TGNConfig(
        neighbor_finder=train_ngh_finder, node_features=node_features, edge_features=edge_features, device=device,
        n_layers=config['n_layer'], n_heads=config['n_head'], dropout=drop_edge_rate, use_memory=use_memory,
        message_dimension=config['message_dim'], memory_dimension=config['memory_dim'],
        memory_update_at_start=config['memory_update_at_start'],
        embedding_module_type=embedding_module, message_function=message_function,
        aggregator_type=aggregator, memory_updater_type=memory_updater,
        n_neighbors=config['n_degree'],
        use_destination_embedding_in_message=use_destination_embedding_in_message,
        use_source_embedding_in_message=use_source_embedding_in_message,
        dyrep=dyrep
    )
    gps_config = GPSConfig(model_name='PatchTST', seq_len=178, pred_len=77, enc_in_endo=1, enc_in_exo=2,  d_model=512,
                           n_heads=8, e_layers=3, d_ff=512, dropout=0.1, fc_dropout=0.2, head_dropout=0.0,
                           do_predict=True, individual=False, patch_len=8, stride=4, padding_patch='end',
                           revin=True, affine=False, subtract_last=False, decomposition=False, kernel_size=8)

    # Initialize Model
    model = DTSModel(vocab_size, route_max_len, road_feat_num, road_embed_size, gps_feat_num,
                      gps_embed_size, route_embed_size, hidden_size, edge_index, drop_edge_rate, drop_route_rate,
                      drop_road_rate, nodes_dict, edge_features_df, tgn_config=tgn_config, gps_config=gps_config, mode='x').cuda()

    # model.node_embedding.weight = torch.nn.Parameter(init_road_emb['init_road_embd'])
    # model.node_embedding.requires_grad_(True)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopMonitor(max_round=5)
    graph_criterion = torch.nn.BCELoss()
    g_criterion = nn.MSELoss(reduction='none')

    # exp information
    nowtime = datetime.now().strftime("%y%m%d%H%M%S")
    model_name = 'DTS_{}_{}_{}_{}_{}'.format(city, version, num_epochs, num_samples, nowtime)
    model_path = os.path.join(save_path, 'DTS_{}_{}'.format(city, nowtime), 'model')
    log_path = os.path.join(save_path, 'DTS_{}_{}'.format(city, nowtime), 'log')

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    checkpoints = [f for f in os.listdir(model_path) if f.startswith(model_name)]
    # writer = SummaryWriter(log_path)
    if not retrain and checkpoints:
        checkpoint_path = os.path.join(model_path, sorted(checkpoints)[-1])
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        model.apply(weight_init)
    total_lists = train_loader.dataset.gps_data.x_time_features_list.shape[0]
    epoch_step = total_lists // batch_size
    total_steps = epoch_step * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=total_steps)
    log_file_path = os.path.join(model_path, "_".join([model_name, "loss_log.txt"]))

    def get_checkpoint_path(epoch):
        return f'{model_path}/-{epoch}.pth'

    for epoch in range(num_epochs):
        per_epoch_train_m_loss = []
        per_epoch_train_g_loss = []
        per_epoch_match_loss = []
        per_epoch_ultimate_loss = []

        per_epoch_mse = []
        per_epoch_mae = []
        per_epoch_mape = []
        per_epoch_rmse = []
        # Reinitialize memory of the model at the start of each epoch
        if tgn_config.use_memory:
            model.tgn.memory.__init_memory__()
        # Train using only training graph
        model.tgn.set_neighbor_finder(tgn_config.neighbor_finder)
        model.train()
        for idx, batch in enumerate(train_loader):
            batch_m_loss = 0.0
            optimizer.zero_grad()
            x_indo_features_batch = batch['x_indo_features'],
            x_exo_features_batch = batch['x_exo_features'],
            x_opath_batch = batch['x_opath'],
            x_time_feature_batch = batch['x_time_feature'],
            y_indo_features_batch = batch['y_indo_features'],
            y_opath_batch = batch['y_opath'],
            y_time_feature_batch = batch['y_time_feature'],

            road_timestamps_batch = batch['road_timestamps']
            gps_length = batch['gps_length']
            sources_batch = batch['sources_batch']
            destinations_batch = batch['destinations_batch']
            edge_idxs_batch = batch['edge_idxs_batch']
            negatives_batch = batch['negatives_batch']
            pos_label = batch['pos_label']
            neg_label = batch['neg_label']
            # 应用 get_middle_timestamps_datetime 函数到 road_timestamps_batch 的每一个内部列表
            road_midl_stamps_batch = [get_middle_timestamps_datetime(ts)[0] for ts in road_timestamps_batch]
            interval = [get_middle_timestamps_datetime(ts)[1] for ts in road_timestamps_batch]

            src_timestamps_batch = [timestamps[:-1]for timestamps in road_midl_stamps_batch]
            src_timestamps_batch = list(itertools.chain.from_iterable(src_timestamps_batch))
            des_timestamps_batch = [timestamps[1:] for timestamps in road_midl_stamps_batch]
            des_timestamps_batch = list(itertools.chain.from_iterable(des_timestamps_batch))
            route_len = [len(timestamps) for timestamps in road_midl_stamps_batch]
            intervals = list(itertools.chain.from_iterable(interval))

            pos_prob, neg_prob, outputs, seq_endogenous_y, y_valid, speed_time, unique_path_ids, truth_interval \
                = model(x_indo_features_batch, x_exo_features_batch, x_opath_batch, x_time_feature_batch,
                        y_indo_features_batch, y_opath_batch, y_time_feature_batch, gps_length,
                        sources_batch, destinations_batch, edge_idxs_batch,
                        src_timestamps_batch, des_timestamps_batch, route_len, intervals,
                        negatives_batch)

            batch_m_loss += graph_criterion(pos_prob.squeeze(), pos_label) + graph_criterion(neg_prob.squeeze(), neg_label)
            per_epoch_train_m_loss.append(batch_m_loss.item())   # 累加损失

            loss_per_point = g_criterion(outputs, seq_endogenous_y)
            y_valid = y_valid.to('cuda:0')
            # 应用掩码，忽略填充值的影响
            masked_loss_per_point = loss_per_point.squeeze() * y_valid
            # 计算总的损失，除以有效元素的数量来获得平均损失
            batch_gps_loss = masked_loss_per_point.sum() / y_valid.sum()
            per_epoch_train_g_loss.append(batch_gps_loss.item())

            average_interval_route = model.tgn.memory.get_memory(unique_path_ids)[:, -128:]
            batch_match_loss, batch_loss_truth, fused_values, truth_interval_tensor \
                = get_road_match_loss(speed_time, average_interval_route, unique_path_ids, truth_interval,
                                      model.tgn.embedding_module.interval_emb, model)
            per_epoch_match_loss.append(batch_match_loss.item())
            per_epoch_ultimate_loss.append(batch_loss_truth.item())

            #  评估
            mse, rmse, mae, mape = compute_losses(fused_values, truth_interval_tensor)
            per_epoch_mse.append(mse)
            per_epoch_mae.append(mae)
            per_epoch_mape.append(mape)
            # per_epoch_mape = remove_outliers(per_epoch_mape)
            per_epoch_rmse.append(rmse)
            avg_train_mse = np.mean(per_epoch_mse)
            avg_train_mae = np.mean(per_epoch_mae)
            avg_train_mape = np.mean(per_epoch_mape)
            avg_train_rmse = np.mean(per_epoch_rmse)

            # # (GRM LOSS) get gps & route rep matching loss
            # tau = 0.07
            # match_loss = get_traj_match_loss(gps_traj_rep, route_traj_rep, model, batch_size, tau)            #
            # # MLM 1 LOSS + MLM 2 LOSS + GRM LOSS
            # loss = (route_mlm_loss + gps_mlm_loss + 2*match_loss) / 3
            # step = epoch_step*epoch + idx

            batch_loss = batch_gps_loss/56 + batch_m_loss/1.4 + batch_match_loss + batch_loss_truth
            batch_loss.backward()
            optimizer.step()
            if tgn_config.use_memory:
                model.tgn.memory.detach_memory()

            if not (idx + 1) % verbose:
                t = datetime.now().strftime('%m-%d %H:%M:%S')
                # print(f'{t} | (Train) | Epoch={epoch}\tbatch_id={idx + 1}\tloss={batch_m_loss.item():.4f}')
                print(f'{t} |(Train)|Epoch={epoch} batch_id={idx + 1} |loss1={batch_gps_loss.item():.4f}\t'
                      f'loss2={batch_m_loss.item():.4f}\tloss3={batch_match_loss.item():.4f}\t'
                      f'loss={batch_loss_truth.item():.4f}\t'
                      f'mae={mae:.4f}\tmape={mape:.4f}')

        scheduler.step()
        # 计算并打印平均损失
        avg_train_m_loss = np.average(per_epoch_train_m_loss)
        # 计算每个 epoch 的平均损失
        avg_train_g_loss = np.average(per_epoch_train_g_loss)
        avg_train_match_loss = np.average(per_epoch_match_loss)
        avg_train_ultimate_loss = np.average(per_epoch_ultimate_loss)


        # Validation uses the full graph
        model.tgn.set_neighbor_finder(full_ngh_finder)
        vali_g_loss, vali_match_loss, vali_ultimate_loss, val_ap, val_auc, \
            aver_epoch_mse, aver_epoch_rmse, aver_epoch_mae, aver_epoch_mape \
            = vali(model, val_loader, val_rand_sampler, tgn_config.n_neighbors,
                   tgn_config.use_memory, g_criterion, log_path, epoch, test=False)

        # log metrics to wandb
        wandb.log({
                   'train_route_loss': avg_train_m_loss,
                   "train_gps_loss": avg_train_g_loss,
                   'train_match_loss': avg_train_match_loss,
                   "train_ultimate_loss": avg_train_ultimate_loss,
                   "train_mae": avg_train_mae,
                   "train_rmse": avg_train_rmse,
                   "train_mape": avg_train_mape,
                   "val_gps_loss": vali_g_loss,
                   'val_match_loss': vali_match_loss,
                   "val_ultimate_loss": vali_ultimate_loss,
                   "val_ap": val_ap,
                   "val_auc": val_auc,
                   "val_rmse": aver_epoch_rmse,
                   "val_mae": aver_epoch_mae,
                   "val_mape": aver_epoch_mape,
                   })
        print(f'Epoch {epoch}, Average Train Loss of GPS: {avg_train_g_loss:.4f},'
              f'| Average Train Loss of Route: {avg_train_m_loss:.4f},'
              f'| Average Train Match Loss: {avg_train_match_loss:.4f},'
              f'| Average Train Ultimate Loss: {avg_train_ultimate_loss:.4f},'
              f'| Average Validate Loss of GPS: {vali_g_loss:.4f},'
              f'| Average Validate Loss of Match: {vali_match_loss:.4f},'
              f'| Average Validate Loss of Ultimate: {vali_ultimate_loss:.4f},'
              f'| val ap: {val_ap}, val auc: {val_auc}'
              f'| Average Validate RMSE: {aver_epoch_rmse:.4f},'
              f'| Average Validate MAE: {aver_epoch_mae:.4f},'
              f'| Average Validate MAPE: {aver_epoch_mape:.4f}'
              )
        # with open(log_file_path, 'a') as log_file:  # 'a' 表示追加模式
        #     log_file.write(f"Epoch {epoch}, Average train Loss: {avg_train_g_loss:.4f}, val ap: {val_ap}, val auc: {val_auc}\n")
        # Early stopping
        # if early_stopper.early_stop_check(vali_g_loss):
        #     # 使用f-string风格的输出
        #     print(f'No improvement over {early_stopper.max_round} epochs, stop training')
        #     print(f'Loading the best model at epoch {early_stopper.best_epoch}')
        #     best_model_path = get_checkpoint_path(early_stopper.best_epoch)
        #     model.load_state_dict(torch.load(best_model_path))
        #     print(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
        #     model.eval()
        #     break
        # else:
        #     # torch.save(model.state_dict(), get_checkpoint_path(epoch))
        #     continue

    # Training has finished, we have loaded the best model, and we want to backup its current
    # memory (which has seen validation edges) so that it can also be used when testing on unseen
    # nodes
    val_memory_backup = model.tgn.memory.backup_memory()

    # Test
    model.tgn.memory.restore_memory(val_memory_backup)
    model.tgn.set_neighbor_finder(full_ngh_finder)
    test_g_loss, test_match_loss, test_ultimate_loss, test_ap, test_auc, test_mse, test_rmse, test_mae, test_mape\
        = vali(model, test_loader, test_rand_sampler, tgn_config.n_neighbors,
               tgn_config.use_memory, g_criterion, log_path, epoch, test=True)
    # log metrics to wandb
    wandb.log({"test_GPS_loss": test_g_loss,
               'test_match_loss': test_match_loss,
               "test_ultimate_loss": test_ultimate_loss,
               "test_ap": test_ap,
               "test_auc": test_auc,
               "test_rmse": test_rmse,
               "test_mae": test_mae,
               "test_mape": test_mape
               })
    print(f'Test: GPS_Loss: {test_g_loss},'
          f'Test: MATCH_Loss: {test_match_loss},'
          f'Test: Ultimate_Loss: {test_ultimate_loss},'
          f'Test: AP: {test_ap}, AUC: {test_auc}'
          f'Test: MSE: {test_mse}, RMSE: {test_rmse}, MAE: {test_mae}, MAPE: {test_mape}'
          )
    # with open(log_file_path, 'a') as log_file:  # 'a' 表示追加模式
    #     log_file.write(f"Test: AP: {test_ap}, AUC: {test_auc}\n")
    wandb.finish()

    # torch.save({
    #     'epoch': epoch,
    #     'model': model,
    #     'optimizer_state_dict': optimizer.state_dict()
    # }, os.path.join(model_path, "_".join([model_name, f'{epoch}.pt'])))

    return model


if __name__ == '__main__':
    config = json.load(open('config/chengdu.json', 'r'))
    print('>>>>>>>start training >>>>>>>>>>>>>>>>>>>>>>>>>>')
    train(config)


