__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from gps_layers.PatchTST_backbone import PatchTST_backbone
from gps_layers.PatchTST_layers import series_decomp


class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        enc_in_endo = configs.enc_in_endo
        enc_in_exo = configs.enc_in_exo
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        self.use_norm = True

        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in_endo=enc_in_endo,c_in_exo=enc_in_exo, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride,
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in_endo=enc_in_endo,c_in_exo=enc_in_exo, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride,
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone(c_in_endo=enc_in_endo,c_in_exo=enc_in_exo, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride,
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
    
    
    def forward(self, x_exo_features_batch, x_indo_features_batch, x_time_feature_batch, x_opath_batch, y_opath_batch, pre_len):
        # x = torch.cat([x_indo_features_batch[0], x_exo_features_batch[0]], dim=2)
        x_indo = x_indo_features_batch[0]  # x: [Batch, Input length, Channel]
        x_exo = x_exo_features_batch[0]
        x_opath_batch = x_opath_batch[0]
        # x = x_indo_features_batch[0]

        if self.use_norm:
            # Normalization from Non-stationary Transformer
            exogenous_means = x_exo.mean(1, keepdim=True).detach()
            x_exo = x_exo - exogenous_means
            exogenous_st = torch.sqrt(torch.var(x_exo, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_exo /= exogenous_st

            endogenous_means = x_indo.mean(1, keepdim=True).detach()
            x_indo = x_indo - endogenous_means
            endogenous_st = torch.sqrt(torch.var(x_indo, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_indo /= endogenous_st

        if self.decomposition:
            res_init, trend_init = self.decomp_module(x_indo)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x_indo = res + trend
            x_indo = x_indo.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x_indo = x_indo.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x_exo = x_exo.permute(0,2,1)
            x_indo = self.model(x_indo, x_exo, x_opath_batch)
            x_indo = x_indo.permute(0,2,1)    # x: [Batch, Input length, Channel]
            if self.use_norm:
                # De-Normalization from Non-stationary Transformer
                x_indo = x_indo.to('cuda:0')
                endogenous_st = endogenous_st.to('cuda:0')
                endogenous_means = endogenous_means.to('cuda:0')
                dec_out = x_indo * (endogenous_st[:, 0, :].unsqueeze(1).repeat(1, pre_len, 1))
                dec_out = dec_out + (endogenous_means[:, 0, :].unsqueeze(1).repeat(1, pre_len, 1))
        return dec_out
