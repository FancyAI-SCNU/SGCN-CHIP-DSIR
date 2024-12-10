import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tools.utils import ZeroSoftmax


# Eq(1) in paper
class SelfAttention(nn.Module):

    def __init__(self, in_dims=2, d_model=64, num_heads=4):
        super(SelfAttention, self).__init__()

        self.embedding = nn.Linear(in_dims, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)

        self.scaled_factor = np.sqrt(d_model)
        self.softmax = nn.Softmax(dim=-1)

        self.num_heads = num_heads

    def split_heads(self, x):

        # x [8, N, 2] or [N, 8, 3]
        x = x.reshape(x.shape[0], -1, self.num_heads, x.shape[-1] // self.num_heads).contiguous()

        return x.permute(0, 2, 1, 3)  # [batch_size nun_heads seq_len depth]

    def forward(self, x, mask=False, multi_head=False):

        # batch_size seq_len 2

        assert len(x.shape) == 3

        embeddings = self.embedding(x)  # batch_size seq_len d_model
        query = self.query(embeddings)  # batch_size seq_len d_model
        key = self.key(embeddings)  # batch_size seq_len d_model

        if multi_head:
            query = self.split_heads(query)  # B num_heads seq_len d_model
            key = self.split_heads(key)  # B num_heads seq_len d_model
            attention = torch.matmul(query, key.permute(0, 1, 3, 2))  # (batch_size, num_heads, seq_len, seq_len)
        else:
            attention = torch.matmul(query, key.permute(0, 2, 1))  # (batch_size, seq_len, seq_len)

        attention = self.softmax(attention / self.scaled_factor)

        if mask is True:
            # if following part is available
            mask = torch.ones_like(attention)
            # print(mask.size(),'---mask', torch.tril(mask).size())
            # print(mask,'---mask', torch.tril(mask))
            attention = attention * torch.tril(mask)  # Returns the lower triangular part of the matrix

        return attention, embeddings


# Discussions above Eq(2)
# 1) stack the dense interactions R^s-t_spa from every time step with size (T_obs,N,N),
# 2) then fuse these stacked interactions with 1x1 convolution along
# the temporal channel, resulting in spatial-temporal dense interactions
class SpatialTemporalFusion(nn.Module):
    # 1x1 convolution to fuse sequential spatial feature maps
    def __init__(self, obs_len=8):
        super(SpatialTemporalFusion, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(obs_len, obs_len, 1),
            nn.PReLU()
        )

        self.shortcut = nn.Sequential()

    def forward(self, x):
        # [num_heads=4, hist_len=8, N, N]
        # Each slice [i, t, :, :] at each time step t is an asymmetric square
        # matrix, where its (i, j)-th element represents the influence
        # of node i to node j
        # print(x.size(),'===============================')
        x = self.conv(x) + self.shortcut(x)
        return x.squeeze()


# Eq (2)
# the input is spatial feature map (hist_len (T_obs) , N, N)
# the output is the high-level interaction feature of the same size
class AsymmetricConvolution(nn.Module):

    def __init__(self, in_cha, out_cha):
        super(AsymmetricConvolution, self).__init__()
        # the in/out-channel is the num_heads in Self-Attention!
        # row and col convolutions
        # Conv2d(4, 4, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False)
        # Conv2d(4, 4, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.conv1 = nn.Conv2d(in_cha, out_cha, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.conv2 = nn.Conv2d(in_cha, out_cha, kernel_size=(1, 3), padding=(0, 1))

        self.shortcut = lambda x: x

        if in_cha != out_cha:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_cha, out_cha, 1, bias=False)
            )

        self.activation = nn.PReLU()

    def forward(self, x):
        shortcut = self.shortcut(x)
        # print(x.size(),'---x')
        # for spatial convs, [hist_len=8, 4, N, N]
        # for temporal convs, [N, 4, hist_len=8, hist_len=9]
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = self.activation(x2 + x1)

        return x2 + shortcut


# Eq(2)-(3)
# generate interaction-mask learned from feature maps
# with elementwise threshold on each interaction element
class InteractionMask(nn.Module):

    def __init__(self, number_asymmetric_conv_layer=7, spatial_channels=4, temporal_channels=4):
        super(InteractionMask, self).__init__()

        # 7-layer spatial-temporal asymmetric conv layers
        self.number_asymmetric_conv_layer = number_asymmetric_conv_layer

        self.spatial_asymmetric_convolutions = nn.ModuleList()
        self.temporal_asymmetric_convolutions = nn.ModuleList()

        for i in range(self.number_asymmetric_conv_layer):
            self.spatial_asymmetric_convolutions.append(
                AsymmetricConvolution(spatial_channels, spatial_channels)
            )
            self.temporal_asymmetric_convolutions.append(
                AsymmetricConvolution(temporal_channels, temporal_channels)
            )

        self.spatial_output = nn.Sigmoid()
        self.temporal_output = nn.Sigmoid()

    def forward(self, dense_spatial_interaction, dense_temporal_interaction, threshold=0.5):

        assert len(dense_temporal_interaction.shape) == 4
        assert len(dense_spatial_interaction.shape) == 4

        for j in range(self.number_asymmetric_conv_layer):
            # print('Spatial',j)
            # [hist_len=8, 4, N, N]
            dense_spatial_interaction = self.spatial_asymmetric_convolutions[j](dense_spatial_interaction)

            # print('Temporal', j)
            # [N, 4, hist_len=8, hist_len=9]
            dense_temporal_interaction = self.temporal_asymmetric_convolutions[j](dense_temporal_interaction)

        # squash the i-j weight to (0,1)
        spatial_interaction_mask = self.spatial_output(dense_spatial_interaction)
        temporal_interaction_mask = self.temporal_output(dense_temporal_interaction)

        spatial_zero = torch.zeros_like(spatial_interaction_mask, device=dense_spatial_interaction.device)
        temporal_zero = torch.zeros_like(temporal_interaction_mask, device=dense_spatial_interaction.device)

        # Eq(3) elementwise threshold
        spatial_interaction_mask = torch.where(spatial_interaction_mask > threshold, spatial_interaction_mask,
                                               spatial_zero)

        temporal_interaction_mask = torch.where(temporal_interaction_mask > threshold, temporal_interaction_mask,
                                                temporal_zero)

        return spatial_interaction_mask, temporal_interaction_mask


# Eq(3)-(5)
# Sparse directed interaction is obtained from all spatial graphs
# the temporal is obtained by concatenating/stacking all spatial graphs along the time axis
class SparseWeightedAdjacency(nn.Module):

    def __init__(self, spa_in_dims=2, tem_in_dims=3, embedding_dims=64, obs_len=8, dropout=0,
                 number_asymmetric_conv_layer=7, num_heads=4):
        super(SparseWeightedAdjacency, self).__init__()

        # dense interaction
        self.spatial_attention = SelfAttention(spa_in_dims, embedding_dims, num_heads=num_heads)
        self.temporal_attention = SelfAttention(tem_in_dims, embedding_dims, num_heads=num_heads)

        # spatial-temporal attention fusion
        self.spa_fusion = SpatialTemporalFusion(obs_len=obs_len)

        # no temporal-spatial fusion

        # interaction mask
        # see Eq(3)
        self.interaction_mask = InteractionMask(
            number_asymmetric_conv_layer=number_asymmetric_conv_layer,
            spatial_channels=num_heads, temporal_channels=num_heads,
        )

        self.dropout = dropout
        self.zero_softmax = ZeroSoftmax()
        self.num_heads = num_heads

    def forward(self, graph, identity):
        assert len(graph.shape) == 3

        spatial_graph = graph[:, :, 1:]  # (T=8 N 2)
        temporal_graph = graph.permute(1, 0, 2)  # (N T=8 3)
        # print(spatial_graph.size(), temporal_graph.size(),'--graph')
        # graph last dim is (id,x,y), the id is 1,2,3,...,8, for creating positional encoding

        # (hist_len=8 num_heads=4 N N)   (T N d_model)
        dense_spatial_interaction, spatial_embeddings = self.spatial_attention(spatial_graph, multi_head=True)

        # (N num_heads T T)   (N T d_model)
        dense_temporal_interaction, temporal_embeddings = self.temporal_attention(temporal_graph, mask=True,
                                                                                  multi_head=True)

        # attention fusion
        # dense_spatial_interaction --- (hist_len=8 num_heads=4 N N)
        # dense_spatial_interaction.permute(1, 0, 2, 3) # -- (4, 8, N, N)
        # self.spa_fusion(dense_spatial_interaction.permute(1, 0, 2, 3) # -- (4, 8, N, N)
        # st_interaction -- (8, 4, N, N)
        # ts_interaction -- (N, 4, 8, 8)
        st_interaction = self.spa_fusion(dense_spatial_interaction.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
        ts_interaction = dense_temporal_interaction
        # print(st_interaction.size(), ts_interaction.size())

        # masking out non-important interaction, Eq(3)
        spatial_mask, temporal_mask = self.interaction_mask(st_interaction, ts_interaction)

        # self-connected, Eq(4)
        spatial_mask = spatial_mask + identity[0].unsqueeze(1)
        temporal_mask = temporal_mask + identity[1].unsqueeze(1)

        # Eq(5)
        '''
        Softmax outputs non-zero values for zero inputs, so the pedestrians that do not interact with each other are forced to interact 
        To avoid this problem, use a “Zero-Softmax“ function to to keep the sparsity
        '''
        normalized_spatial_adjacency_matrix = self.zero_softmax(dense_spatial_interaction * spatial_mask, dim=-1)
        normalized_temporal_adjacency_matrix = self.zero_softmax(dense_temporal_interaction * temporal_mask, dim=-1)

        return normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix, \
               spatial_embeddings, temporal_embeddings


# Eq(6)
class GraphConvolution(nn.Module):

    def __init__(self, in_dims=2, embedding_dims=16, dropout=0):
        super(GraphConvolution, self).__init__()

        self.embedding = nn.Linear(in_dims, embedding_dims, bias=False)  # Eq(6) W parameter
        self.activation = nn.PReLU()

        self.dropout = dropout

    def forward(self, graph, adjacency, verbose=False):
        # graph [batch_size 1 seq_len 2]
        # adjacency [batch_size num_heads seq_len seq_len]
        # the adjacency is from above normalized_spatial/temporal_adjacency matrix
        # they sum to one in row wise
        if verbose:
            print(adjacency.size(), graph.size())
            tmp = torch.matmul(adjacency, graph)
            print(tmp.size())
        gcn_features = self.embedding(torch.matmul(adjacency, graph))
        gcn_features = F.dropout(self.activation(gcn_features), p=self.dropout)

        return gcn_features  # [batch_size num_heads seq_len hidden_size]


# Fig.2 GCNs
# Sec 3.2 with Eq(6)
# use two GCNs to learn the trajectory representation,
# where in one branch A_spa is fed to the network ahead of A_tmp,
# while in the other branch they are fed in the reverse order.
class SparseGraphConvolution(nn.Module):

    def __init__(self, in_dims=16, embedding_dims=16, dropout=0):
        super(SparseGraphConvolution, self).__init__()

        self.dropout = dropout

        self.spatial_temporal_sparse_gcn = nn.ModuleList()
        self.temporal_spatial_sparse_gcn = nn.ModuleList()

        self.spatial_temporal_sparse_gcn.append(GraphConvolution(in_dims, embedding_dims))
        self.spatial_temporal_sparse_gcn.append(GraphConvolution(embedding_dims, embedding_dims))

        self.temporal_spatial_sparse_gcn.append(GraphConvolution(in_dims, embedding_dims))
        self.temporal_spatial_sparse_gcn.append(GraphConvolution(embedding_dims, embedding_dims))

    def forward(self, graph, normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix):
        # graph [1 seq_len num_pedestrians  3]
        # _matrix [batch num_heads seq_len seq_len]

        graph = graph[:, :, :, 1:]
        spa_graph = graph.permute(1, 0, 2, 3)  # (seq_len 1 num_p 2)
        tem_graph = spa_graph.permute(2, 1, 0, 3)  # (num_p 1 seq_len 2)

        ##########################
        # spatial_temporal branch
        ##########################
        # normalized_spatial_adjacency_matrix [hist, n_head=4, N,N]
        # fed spatial adjacency matrix first, then temporal adjacency
        gcn_spatial_features = self.spatial_temporal_sparse_gcn[0](spa_graph, normalized_spatial_adjacency_matrix)
        gcn_spatial_features = gcn_spatial_features.permute(2, 1, 0, 3)
        ''' print(normalized_spatial_adjacency_matrix[0, 0, ...])
        [[0.6892, 0.1055, 0.1045, 0.1008],
        [0.1282, 0.6525, 0.1085, 0.1107],
        [0.1294, 0.1134, 0.6464, 0.1108],
        [0.1250, 0.1086, 0.0977, 0.6686]]
        '''
        # gcn_spatial_features is [N_ped num_heads=4 seq_len=8 d=16]
        # normalized_temporal_adjacency_matrix is [N_ped, 4, 8, 8]
        # the gcn makes a torch mult [N, nh, 8, 8] * [N, nh, 8, 16] -> [N, nh, 8, 16]
        # print(gcn_spatial_features.size(),normalized_temporal_adjacency_matrix.size())
        gcn_spatial_temporal_features = self.spatial_temporal_sparse_gcn[1](gcn_spatial_features,
                                                                            normalized_temporal_adjacency_matrix, False)
        # print(gcn_spatial_temporal_features.size())
        ##########################
        # temporal_spatial branch
        ##########################
        # normalized_temporal_adjacency_matrix [N, n_head=4, hist=8, hist=8]
        # it's a lower-part triangular matrix, with each row sums to 1
        '''print(normalized_temporal_adjacency_matrix[0,0,...])
        [[0.9999, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.1434, 0.8565, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.1570, 0.1535, 0.6894, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.1980, 0.1683, 0.1231, 0.5105, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2630, 0.1985, 0.1268, 0.0859, 0.3258, 0.0000, 0.0000, 0.0000],
        [0.3411, 0.2298, 0.1316, 0.0781, 0.0486, 0.1706, 0.0000, 0.0000],
        [0.4219, 0.2515, 0.1287, 0.0681, 0.0369, 0.0215, 0.0713, 0.0000],
        [0.5038, 0.2469, 0.1159, 0.0575, 0.0289, 0.0154, 0.0088, 0.0228]],
        '''
        # [N, 4, 8, 16]-> permute to  [hist_len=8, num_heads=4, N, d=16]
        # fed temporal adjacency matrix first, then spatial adjacency
        gcn_temporal_features = self.temporal_spatial_sparse_gcn[0](tem_graph,
                                                                    normalized_temporal_adjacency_matrix)
        gcn_temporal_features = gcn_temporal_features.permute(2, 1, 0, 3)

        # gcn_temporal_features [8, 4, N, 16]
        # normalized_spatial_adjacency_matrix [8, 4, N, N]
        # the gcn makes a torch mult [8, 4, N, N] * [8, 4, N, 16] -> [8, 4, N, 16]
        # print(gcn_temporal_features.size(),normalized_spatial_adjacency_matrix.size(),'--')
        gcn_temporal_spatial_features = self.temporal_spatial_sparse_gcn[1](gcn_temporal_features,
                                                                            normalized_spatial_adjacency_matrix, False)

        return gcn_spatial_temporal_features, gcn_temporal_spatial_features.permute(2, 1, 0, 3)


class TrajectoryModel(nn.Module):

    def __init__(self,
                 number_asymmetric_conv_layer=7, embedding_dims=64, number_gcn_layers=1, dropout=0,
                 obs_len=8, pred_len=12, n_tcn=5,
                 out_dims=5, num_heads=4):
        super(TrajectoryModel, self).__init__()

        self.number_gcn_layers = number_gcn_layers
        self.n_tcn = n_tcn
        self.dropout = dropout

        # sparse graph learning
        self.sparse_weighted_adjacency_matrices = SparseWeightedAdjacency(
            number_asymmetric_conv_layer=number_asymmetric_conv_layer,
            num_heads=num_heads
        )

        # graph convolution
        self.stsgcn = SparseGraphConvolution(
            in_dims=2, embedding_dims=embedding_dims // num_heads, dropout=dropout
        )

        self.fusion_ = nn.Conv2d(num_heads, num_heads, kernel_size=1, bias=False)

        '''
        Time Convolution Network (TCN)  predict the parameters of a bi-Gaussian distribution, which
        generates the predicted trajectory. TCN is chosen because it does not suffer from gradient vanishing and 
        high computational cost like RNN
        '''
        self.tcns = nn.ModuleList()
        self.tcns.append(nn.Sequential(
            nn.Conv2d(obs_len, pred_len, 3, padding=1),
            nn.PReLU()
        ))
        for j in range(1, self.n_tcn):
            self.tcns.append(nn.Sequential(
                nn.Conv2d(pred_len, pred_len, 3, padding=1),
                nn.PReLU()
            ))

        self.output = nn.Linear(embedding_dims // num_heads, out_dims)

        # CLIP temperature
        # See CLIP paper Sec. 2.5
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        hidden_size = 16
        self.rel_enc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.PReLU(),
        )

        self.unary_enc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.PReLU(),
        )

        self.out_dist = nn.Linear(hidden_size, 1)
        self.out_unary = nn.Linear(hidden_size, 1)

    def forward(self, graph, identity, use_train=True):

        # graph 1 obs_len N 3

        # Eq(3)-(5)
        # spatial and temporal adjacency learning
        # the importance between i-j pair
        normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix, spatial_embeddings, temporal_embeddings = \
            self.sparse_weighted_adjacency_matrices(graph.squeeze(), identity)

        # Eq(6)-(7)
        # sparse GCN
        # gcn_temporal_spatial_features, gcn_spatial_temporal_features [N, 4, 8, 16]
        gcn_temporal_spatial_features, gcn_spatial_temporal_features = self.stsgcn(
            graph, normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix
        )

        gcn_representation = self.fusion_(gcn_temporal_spatial_features) + gcn_spatial_temporal_features

        # [N, hist_len=8, num_heads=4, d=16]
        gcn_representation = gcn_representation.permute(0, 2, 1, 3).contiguous()

        # [N, fut_len=12, 4, 16]
        features = self.tcns[0](gcn_representation)

        for k in range(1, self.n_tcn):
            features = F.dropout(self.tcns[k](features) + features, p=self.dropout)

        # bi-Gaussian prediction to dim=5
        # [N, fut_len=12, 4, 5]
        pred_1 = self.output(features)
        # [N, 12, 5], average over all heads
        pred_2 = torch.mean(pred_1, dim=-2)
        # [N, 12, 5] -> [12, N, 5]
        prediction = pred_2.permute(1, 0, 2).contiguous()

        if not use_train:
            return prediction

        # [N, 8, 64]
        feature_enc = gcn_representation.view(gcn_representation.size(0), gcn_representation.size(1), -1)
        # [N, 12, 64]
        feature_dec = features.view(features.size(0), features.size(1), -1)

        # clip
        # [N, 8, hidden] -> [N, hidden]
        out_hist = torch.mean(feature_enc, dim=-2)
        # [N, 12, hidden] -> [N, hidden]
        out_future = torch.mean(feature_dec, dim=-2)
        # normalized features
        # https://github.com/openai/CLIP/blob/main/clip/model.py#L368
        # out_hist = out_hist / out_hist.norm(dim=1, keepdim=True)
        # out_future = out_future / out_future.norm(dim=1, keepdim=True)
        logits_hist_future = torch.matmul(out_hist, out_future.T)

        return prediction, logits_hist_future
