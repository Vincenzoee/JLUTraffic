__all__ = ['PatchTST_backbone']

# Cell
from typing import Callable, Optional
import torch
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import seaborn as sns
#from collections import OrderedDict
from gps_layers.PatchTST_layers import *
from gps_layers.RevIN import RevIN


def create_patch_mask(x_opath_batch, patch_size=8, stride=4):
    # Step 1: 标记出所有的NaN值
    nan_mask = torch.eq(x_opath_batch, -1.0)

    # Step 2: 初始化mask矩阵
    batch_size, sequence_length = x_opath_batch.shape
    num_patches = (sequence_length - patch_size) // stride + 2
    patch_mask = torch.zeros(batch_size, num_patches).bool()  # 假设我们只需要bool类型的mask

    # Step 3 & 4: 计算每个patch中的NaN比例，并根据比例更新mask矩阵
    for b in range(batch_size):
        for p in range(num_patches):
            start_idx = p * stride
            end_idx = start_idx + patch_size
            patch_nan_count = nan_mask[b, start_idx:end_idx].sum()
            if patch_nan_count >= patch_size / 2:  # 如果一半以上的点是NaN，则标记为1
                patch_mask[b, p] = True  # 标记为True表示1

    return patch_mask


# Cell
class PatchTST_backbone(nn.Module):
    def __init__(self, c_in_endo:int,c_in_exo:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024,
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                 verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # RevIn
        self.revin = revin
        if self.revin: 
            self.revin_layer = RevIN(c_in_endo, affine=affine, subtract_last=subtract_last)
            self.revin_layer_exo = RevIN(c_in_exo, affine=affine, subtract_last=subtract_last)
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1
        
        # Backbone 
        self.backbone = TSTiEncoder(c_in_endo, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in_endo
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, c_in_endo, fc_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten': 
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
        
    
    def forward(self, z, x_exo, x_opath_batch):                                           # z: [bs x nvars x seq_len]
        # norm
        if self.revin: 
            z = z.permute(0,2,1)
            x_exo = x_exo.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            x_exo = self.revin_layer_exo(x_exo, 'norm')
            z = z.permute(0,2,1)
            x_exo = x_exo.permute(0,2,1)
            
        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
            x_exo = self.padding_patch_layer(x_exo)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        x_exo = x_exo.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]
        x_exo = x_exo.permute(0,1,3,2)
        # 使用函数
        patch_mask = create_patch_mask(x_opath_batch, self.patch_len, self.stride)

        # model
        z = self.backbone(z, x_exo, patch_mask)                                             # z: [bs x nvars x d_model x patch_num]
        z = self.head(z)                                                                    # z: [bs x nvars x target_window] 
        
        # denorm
        if self.revin: 
            z = z.permute(0,2,1)
            z = z.to('cuda:0')
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        return z
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))  # d_model * patch_num
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x
        
        
    
    
class TSTiEncoder(nn.Module):  #i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):
        
        
        super().__init__()
        
        self.patch_num = patch_num
        self.patch_len = patch_len
        
        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.W_P_exo = nn.Linear(patch_len, d_model)    # Eq 1: projection of feature vectors onto a d-dim vector space

        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)
        self.W_pos_exo = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)

        
    def forward(self, x, x_exo, patch_mask) -> Tensor:                           # x: [bs x nvars x patch_len x patch_num]
        
        n_vars_endo = x.shape[1]
        n_vars_exo = x_exo.shape[1]

        # Input encoding
        x = x.permute(0,1,3,2)                                                   # x: [bs x nvars x patch_num x patch_len]
        x = x.to(self.W_P.weight.device)
        x = x.float()
        x = self.W_P(x)                                                          # x: [bs x nvars x patch_num x d_model]

        x_exo = x_exo.permute(0,1,3,2)                                          # x: [bs x nvars x patch_num x patch_len]
        x_exo = x_exo.to(self.W_P_exo.weight.device)
        x_exo = x_exo.float()
        x_exo = self.W_P_exo(x_exo)                                                  # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))      # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x patch_num x d_model]
        patch_mask_endo = patch_mask.repeat(n_vars_endo, 1)

        # exo = torch.reshape(x_exo, (x_exo.shape[0]*x_exo.shape[1],x_exo.shape[2],x_exo.shape[3]))
        exo = self.dropout(x_exo + self.W_pos_exo)
        patch_mask_exo = patch_mask.unsqueeze(1).repeat(1, n_vars_exo, 1)

        # Encoder
        z = self.encoder(u, exo, patch_mask_endo, patch_mask_exo)                # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(z, (-1,n_vars_endo,z.shape[-2],z.shape[-1]))           # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]
        
        return z    
            
            
    
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor, exo:Tensor,key_padding_mask:Optional[Tensor]=None, patch_mask_exo:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output_endo = src
        scores_endo = None
        output_exo = exo
        scores_exo = None
        if self.res_attention:
            for mod in self.layers: output_endo, scores_endo, output_exo, scores_exo = \
                mod(output_endo, prev_self=scores_endo, key_padding_mask=key_padding_mask,
                    exo=output_exo, prev_cross=scores_exo, patch_mask_exo=patch_mask_exo, attn_mask=attn_mask)
            return output_endo
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output



class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)
        if "batch" in norm.lower():
            self.norm_attn_exo = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn_exo = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev_self:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None,
                exo:Optional[Tensor]=None, prev_cross:Optional[Tensor]=None, patch_mask_exo:Optional[Tensor]=None,
                attn_mask:Optional[Tensor]=None) -> Tensor:
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
            exo = self.norm_attn_exo(exo)
        # Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev_self, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        # Add & Norm
        src = src + self.dropout_attn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)
        # 交叉注意力
        if self.res_attention:
            exo2, attn_exo, scores_exo = self.self_attn(src, exo, exo, prev_cross, key_padding_mask=patch_mask_exo, attn_mask=attn_mask)
            # 合并三个外生变量的影响
            exo3 = exo2.sum(dim=1)  # context : [bs x q_len x n_heads*d_v]
        else:
            exo2, attn_exo, scores_exo = self.self_attn(src, exo, exo, key_padding_mask=patch_mask_exo, attn_mask=attn_mask)
            # 合并三个外生变量的影响
            exo3 = exo2.sum(dim=1)  # context : [bs x q_len x n_heads*d_v]
        if self.store_attn:
            self.attn_exo = attn_exo
        ## Add & Norm
        src = src + self.dropout_attn(exo3)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn_exo(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores, exo2, scores_exo
        else:
            return src




class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if Q.shape == K.shape:
            cross = False
            # Linear (+ split in multiple heads)
            q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
            k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
            v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

            # Apply Scaled Dot-Product Attention (multiple heads)
            if self.res_attention:
                output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, cross, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            else:
                output, attn_weights = self.sdp_attn(q_s, k_s, v_s,  key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

            # back to the original inputs dimensions
            output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
            output = self.to_out(output)
        else:
            exo_var = K.shape[1]
            cross = True
            q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)  # q_s    : [bs x n_heads x max_q_len x d_k]
            k_s = self.W_K(K).view(bs, exo_var, -1, self.n_heads, self.d_k).permute(0, 3, 1, 2, 4)  # k_s : [bs x n_heads x 3 x q_len x d_k]
            v_s = self.W_V(V).view(bs, exo_var, -1, self.n_heads, self.d_v).permute(0, 3, 1, 2, 4)  # v_s : [bs x n_heads x 3 x q_len x d_v]
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, cross, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa
        self.n_heads = n_heads

    def forward(self, q:Tensor, k:Tensor, v:Tensor, cross:bool, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''
        if not cross:
            # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
            attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

            # Add pre-softmax attention scores from the previous layer (optional)
            if prev is not None: attn_scores = attn_scores + prev

            # Attention mask (optional)
            if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
                if attn_mask.dtype == torch.bool:
                    attn_scores.masked_fill_(attn_mask, -np.inf)
                else:
                    attn_scores += attn_mask

            # Key padding mask
            if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
                attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)
            # Apply key padding mask to attention scores
            if key_padding_mask is not None:
                attn_mask_key = key_padding_mask.unsqueeze(1).unsqueeze(2)  # attn_mask_key: [bs * nvars x 1 x 1 x patch_num]
                attn_mask_key = attn_mask_key.repeat(1, self.n_heads, 1, 1)  # attn_mask_key: [bs * nvars x n_heads x 1 x patch_num]
                attn_mask_key = attn_mask_key.to('cuda:0')

                attn_scores = attn_scores.masked_fill(attn_mask_key == 1, float('-inf'))  # Apply key mask

            # normalize the attention weights
            attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
            attn_weights = self.attn_dropout(attn_weights)

            # compute the new values given the attention weights
            output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]
        else:
            attn_scores = torch.matmul(q.unsqueeze(2), k.permute(0, 1, 2, 4, 3))  # attn_weights : [bs x n_heads x 3 x q_len x q_len]
            # Add pre-softmax attention scores from the previous layer (optional)
            if prev is not None: attn_scores = attn_scores + prev
            # Apply key padding mask to attention scores
            if key_padding_mask is not None:
                q_len = attn_scores.shape[-1]
                attn_mask_key = key_padding_mask.unsqueeze(1).unsqueeze(4).repeat(1, self.n_heads, 1, 1, q_len)  # [bs x n_heads x 3 x q_len x q_len]
                attn_mask_key = attn_mask_key.to('cuda:0')

                attn_scores = attn_scores.masked_fill(attn_mask_key == 1, float('-1e8'))  # Apply key mask

            # normalize the attention weights
            attn_weights = F.softmax(attn_scores, dim=-1)  # attn_weights   : [bs x n_heads x max_q_len x q_len]
            # # 绘图
            # batch_idx = 0  # 选择批次中的第一个样本
            # head_idx = 0  # 选择第一个头
            # # 提取特定样本和头的注意力权重，并转换为numpy数组
            # attention_matrix = attn_weights[batch_idx, head_idx, 0].cpu().detach().numpy()
            # # 绘制热力图
            # plt.figure(figsize=(10, 8))
            # sns.heatmap(attention_matrix, annot=False, cmap='viridis', cbar=True)
            # plt.title(f'Attention Weights Heatmap (Batch {batch_idx}, Head {head_idx})')
            # plt.xlabel('Query Position')
            # plt.ylabel('Key Position')
            # plt.show()

            attn_weights = self.attn_dropout(attn_weights)
            # compute the new values given the attention weights
            output = torch.matmul(attn_weights, v)  # output: [bs x n_heads x 3 x max_q_len x d_v]
            # 将上下文向量重新组合成原始形状
            context = output.permute(0, 2, 1, 3, 4)
            context = context.contiguous().view(output.shape[0], output.shape[2], output.shape[3],
                                                                       -1)  # context : [bs x 3 x q_len x n_heads*d_v]

            output = context

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

