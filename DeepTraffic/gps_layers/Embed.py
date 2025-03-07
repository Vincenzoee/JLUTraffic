import bisect
import torch.nn.functional as F
import torch
import torch.nn as nn
import math

from torch.nn.utils.rnn import pad_sequence


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        # c_in 在iTransformer中直接是configs.seq_len
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) 
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class CustomPatchEmbedding(nn.Module):
    def __init__(self, d_model, dropout, patch_lengths=None):
        super(CustomPatchEmbedding, self).__init__()

        # Projection layers to embed the patches into d_model dimensions
        if patch_lengths is None:
            self.patch_split_lengths = [5, 10, 17, 24]
        self.value_embeddings = nn.ModuleList([
            nn.Linear(length, d_model, bias=False) for length in patch_lengths
        ])

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_opath_batch):
        # x is the input data tensor, shape [batch_size, seq_length, features]
        # x_opath_batch contains the segment IDs for each point in the sequence
        batch_size, n_vars, _ = x.shape
        # Initialize mask matrix
        mask = x_opath_batch == -1  # Mask for padding segments
        all_patches = []
        all_masks = []

        for i in range(batch_size):
            current_segment = []
            current_mask = []
            last_segment_id = None

            for j in range(n_vars):
                segment_id = x_opath_batch[i][j].item()

                # Check if this point should be added to the current segment
                if segment_id != -1:
                    if last_segment_id is None or segment_id == last_segment_id:
                        current_segment.append(x[i][j])
                        current_mask.append(False)  # False means not masked
                    else:
                        # New segment starts here, process the old one
                        if len(current_segment) > 0:
                            all_patches.append((current_segment, i))  # Store the segment and its row index
                            all_masks.append((current_mask, i))

                        # Start new segment
                        current_segment = [x[i][j]]
                        current_mask = [False]
                    last_segment_id = segment_id
                else:
                    # Handle padding case
                    if len(current_segment) > 0:
                        all_patches.append((current_segment, i))  # Store the segment and its row index
                        all_masks.append((current_mask, i))

                    # Since we hit a -1, which indicates end of valid data,
                    # break out of the loop and move to the next sample.
                    break

            # Process the last segment if it exists
            if len(current_segment) > 0:
                all_patches.append((current_segment, i))  # Store the segment and its row index
                all_masks.append((current_mask, i))

        padded_patches = []
        padded_masks = []

        for patch, mask in zip(all_patches, all_masks):
            patch, row_index = patch  # Unpack the patch and its row index
            mask, _ = mask  # Unpack the mask and its row index (not used here)

            length = len(patch)
            # 找到最接近的分割长度
            closest_length = min(self.patch_split_lengths, key=lambda x: abs(x - length))
            # 对patch进行填充
            padded_patch = F.pad(torch.tensor(patch), (0, closest_length - length), value=0)
            padded_mask = F.pad(torch.tensor(mask), (0, closest_length - length), value=True)

            padded_patches.append((padded_patch, row_index))  # Store the padded patch and its row index
            padded_masks.append((padded_mask, row_index))  # Store the padded mask and its row index

        # Sort the padded patches and masks by row index
        padded_patches = [patch for patch, _ in sorted(padded_patches, key=lambda x: x[1])]
        padded_masks = [mask for mask, _ in sorted(padded_masks, key=lambda x: x[1])]

        # Project the padded patches using the embedding layers
        embedded_patches = []
        for padded_patch, _ in padded_patches:
            length = padded_patch.shape[0]
            embedding_layer = self.value_embeddings[bisect.bisect_right(self.patch_split_lengths, length) - 1]
            embedded_patch = embedding_layer(padded_patch)
            embedded_patches.append(embedded_patch)

        # Concatenate the embedded patches for each row
        batch_embedded_data = []
        for i in range(batch_size):
            row_patches = [embedded_patches[j] for j in range(len(embedded_patches)) if padded_patches[j][1] == i]
            concatenated_patch = torch.cat(row_patches, dim=0)
            batch_embedded_data.append(concatenated_patch)

        # Stack all embedded data
        embedded_data = torch.stack(embedded_data)

        # Add positional embeddings
        embedded_data += self.position_embedding(embedded_data)

        # Apply dropout
        output = self.dropout(embedded_data)

        # Stack masks
        mask_tensor = torch.stack(
            [torch.cat([m, torch.ones(max_patch_lengths[len(m)] - len(m), dtype=torch.bool)], dim=0) for m in
             all_masks])

        return output, mask_tensor
# class PatchEmbedding(nn.Module):
#     def __init__(self, d_model, patch_len, stride, padding, dropout):
#         super(PatchEmbedding, self).__init__()
#         # Patching
#         self.patch_len = patch_len
#         self.stride = stride
#         self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
#
#         # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
#         self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
#
#         # Positional embedding
#         self.position_embedding = PositionalEmbedding(d_model)
#
#         # Residual dropout
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x, x_opath_batch):
#         # do patching
#         n_vars = x.shape[1]
#         # print(x.shape)
#         # torch.Size([16, 96, 1])
#         x = self.padding_patch_layer(x)
#         # print(x.shape)
#         # torch.Size([16, 96, 9])
#         x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # 将向量分patch
#         # print(x.shape)
#         x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
#         # Input encoding
#         x = self.value_embedding(x) + self.position_embedding(x)
#         return self.dropout(x), n_vars
