import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_atten_mask(graph_nodes):
    batch_size = graph_nodes.size(0)
    seq_len = graph_nodes.size(1)

    pad_attn_mask = graph_nodes.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)

class Time2vec(nn.Module):
    def __init__(self, temporal_dim):
        super(Time2vec, self).__init__()
        self.temporal_dim = temporal_dim
        self.linear_trans = nn.Linear(1, temporal_dim // 2)
        self.cos_trans = nn.Linear(1, temporal_dim // 2)

    def forward(self, t):
        ta = self.linear_trans(t)
        tb = self.cos_trans(t)
        te = torch.cat([ta, tb], -1)
        return te


class PredictionLayer(nn.Module):
    def __init__(self, hidden_dim, dropout_rate):
        super(PredictionLayer, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.linear2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.linear3 = nn.Linear(hidden_dim // 4, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(x)
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.relu(self.linear2(x))
        x = self.linear3(self.dropout(x))
        return x


class CTformer(pl.LightningModule):
    def __init__(self, N, m, hidden_dim, temporal_dim,
                 num_heads, dropout_rate, attn_dropout_rate, ffn_dim,
                 num_layers, lr, weight_decay, alpha, beta, lr_decay_step,
                 lr_decay_gamma, LPE, TE, SPE, TIE, LCA, SD_A, SD_B):
        super(CTformer, self).__init__()
        self.num_heads = num_heads
        self.lr = lr
        self.weight_decay = weight_decay
        self.alpha, self.beta = alpha, beta
        self.lr_decay_step, self.lr_decay_gamma = lr_decay_step, lr_decay_gamma
        self.LPE, self.TE, self.SPE, self.TIE, self.LCA, self.SD_A, self.SD_B = LPE, TE, SPE, TIE, LCA, SD_A, SD_B

        # initial node features (N, d)
        self.user_encoder = nn.Embedding(N, hidden_dim, padding_idx=0)
        # laplacian positional encoding
        if self.LPE:
            self.lpe_encoder = nn.Linear(m, hidden_dim)
        # shortest path length encoding
        if self.SPE:
            self.spl_encoder = nn.Embedding(64, num_heads, padding_idx=0)
        # relative temporal bias
        if self.TIE:
            self.td_encoder = nn.Linear(1, num_heads)
        # LCA bias
        if self.LCA:
            self.lca_encoder = nn.Linear(hidden_dim, num_heads)
        # time embedding
        if self.TE:
            self.te = Time2vec(temporal_dim)
        # stack Transformer layers
        self.graphEncoderLayers = nn.ModuleList(
            [GraphEncoderLayer(hidden_dim, num_heads, dropout_rate, attn_dropout_rate, ffn_dim)
             for _ in range(num_layers)])
        # prediction layernorm and MLPs
        self.f_layernorm = nn.LayerNorm(hidden_dim)
        self.prediction = PredictionLayer(hidden_dim, dropout_rate)
        # Loss: MSE + SAD + SBD
        self.ctloss = CTLoss()
        self.save_hyperparameters()

    def forward(self, batch_data):
        """
        graph_nodes: (batch, seq_len),
        temporal_lst: (batch, seq_len),
        lpe: (batch, seq_len, m),
        spl_matrix: (batch, seq_len, seq_len),
        lca_matrix: (batch, seq_len, seq_len),
        interval_matrix: (batch, seq_len, seq_len),
        labels: (batch, 1),
        real_len: (batch, 1),
        """
        graph_nodes, temporal_lst, lpe, spl_matrix, interval_matrix, lca_matrix, labels, real_len = batch_data
        batch_size, seq_len = graph_nodes.size()
        temporal_lst = temporal_lst.unsqueeze(-1)  # (batch, seq_len, 1)
        x = self.user_encoder(graph_nodes)
        if self.LPE:
            # random sign flipping
            sign_filp = torch.rand(lpe.size(2)).to(x.device)
            sign_filp[sign_filp >= 0.5] = 1.0
            sign_filp[sign_filp < 0.5] = -1.0
            lpe = lpe * sign_filp.unsqueeze(0).unsqueeze(0)
            # add linear_trans to lpe to combine the multiplicities of eig vectors.
            lpe_trans = self.lpe_encoder(lpe)  # add linear transformation to lpe
            x += lpe_trans  # add lpe encoding to nodes
        if self.TE:
            x += self.te(temporal_lst)  # (batch, seq_len, d), add temporal encoding to nodes
        ########################
        # computing attention bias
        attn_bias = torch.zeros((batch_size, seq_len, seq_len, self.num_heads), device=x.device)
        if self.SPE:
            spl_bias = self.spl_encoder(spl_matrix)  # (batch, seq_len, seq_len, num_heads)
            attn_bias += spl_bias
        if self.TIE:
            interval_bias = torch.cos(
                self.td_encoder(interval_matrix.unsqueeze(-1)))  # (batch, seq_len, seq_len, num_heads)
            attn_bias += interval_bias
        if self.LCA:
            # lca_bias item
            bz, n = graph_nodes.size()
            aux_source_idx, aux_lca_idx, aux_target_idx = lca_matrix[:,:,:n], lca_matrix[:,:,n:2*n], lca_matrix[:,:,2*n:]
            source_vectors, target_vectors, lca_vectors = F.softmax(self.user_encoder(aux_source_idx), -1), F.softmax(
                 self.user_encoder(aux_target_idx), -1), F.softmax(self.user_encoder(aux_lca_idx), -1)
            lca_bias =1 - (F.kl_div(lca_vectors.log(), source_vectors, reduction='none') + F.kl_div(target_vectors.log(),
                                                                                                 lca_vectors,
                                                                                                 reduction='none'))
            lca_bias = self.lca_encoder(lca_bias)  # (batch, seq_len, seq_len, num_heads)
            attn_bias += lca_bias

        attn_bias = attn_bias.transpose(1, 3).contiguous()   # (batch, num_heads, seq_len, seq_len)
        ########################
        attn_lst = []
        batch_relations = []
        attn_mask = get_atten_mask(graph_nodes)
        for encoder in self.graphEncoderLayers:
            x, attn = encoder(x, attn_bias, attn_mask)
            bx = x.sum(1)
            relation = F.softmax(torch.matmul(bx, bx.transpose(0, 1)) * (bx.size(1) ** -0.5), -1)
            attn_lst.append(attn)
            batch_relations.append(relation)
        # sum pooling
        x = x.sum(1)  # (batch_size, d)
        x = self.f_layernorm(x)
        out = self.prediction(x)
        return out, attn_lst, batch_relations, x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.StepLR(
                                      optimizer,
                                      step_size=self.lr_decay_step,
                                      gamma=self.lr_decay_gamma),
                        'interval': 'epoch'}
        return [optimizer], [lr_scheduler]

    def training_step(self, batch_data, batch_idx):
        y_pre, attn_lst, batch_relations, _ = self(batch_data)
        tgt = batch_data[-2].view(-1, 1)
        loss = self.ctloss(y_pre, tgt, 1, attn_lst, batch_relations, self.alpha, self.beta, self.SD_A, self.SD_B)
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch_data, batch_idx):
        y_pre, _, _, _ = self(batch_data)
        tgt = batch_data[-2].view(-1, 1)
        return {'y_pre': y_pre, 'tgt': tgt}

    def validation_epoch_end(self, outputs):
        y_pre = torch.cat([o['y_pre'] for o in outputs])
        tgt = torch.cat([o['tgt'] for o in outputs])
        loss = self.ctloss(y_pre, tgt)
        self.log('valid_loss', loss, sync_dist=True)

    def test_step(self, batch_data, batch_idx):
        y_pre, _, _, _ = self(batch_data)
        tgt = batch_data[-2].view(-1, 1)
        return {'y_pre': y_pre, 'tgt': tgt}

    def test_epoch_end(self, outputs):
        y_pre = torch.cat([o['y_pre'] for o in outputs])
        tgt = torch.cat([o['tgt'] for o in outputs])
        result = y_pre.cpu().float().numpy()
        real = tgt.cpu().float().numpy()
        torch.save(np.vstack((result, real)), 'res.pt')
        loss = self.ctloss(y_pre, tgt)
        self.log('test_loss', loss, sync_dist=True)

    @staticmethod
    def setting_model_args(parent_parser):
        parser = parent_parser.add_argument_group("CTformer")
        parser.add_argument('--hidden_dim', type=int, default=32, help="The hidden dimension of models.")
        parser.add_argument('--ffn_dim', type=int, default=32, help="The hidden dimension of FFN layers.")
        parser.add_argument('--temporal_dim', type=int, default=32, help="Time embedding dimension.")
        parser.add_argument('--num_heads', type=int, default=8, help='The num heads of attention.')
        parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout prob.')
        parser.add_argument('--attn_dropout_rate', type=float, default=0.1, help='Attention dropout prob.')
        parser.add_argument('--num_layers', type=int, default=6, help="Num of Transformer encoder layers.")
        parser.add_argument('--lr', type=float, default=0.005, help='Learning rate.')
        parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay.')
        parser.add_argument('--m', type=int, default=32, help="The smallest m eigenvalues.")
        parser.add_argument('--batch_size', type=int, default=4, help="Batch size.")
        parser.add_argument('--alpha', type=float, default=1, help="Self attention scores distillation loss rate.")
        parser.add_argument('--beta', type=float, default=1, help="Batch attention scores distillation loss rate.")
        parser.add_argument('--clip_val', type=float, default=5.0, help="Gradient clipping values. ")
        parser.add_argument('--total_epochs', type=int, default=100, help="Max epochs of model training.")
        parser.add_argument('--lr_decay_step', type=int, default=25, help="Learning rate decay step size.")
        parser.add_argument('--lr_decay_gamma', type=float, default=0.5, help="Learning rate decay rate.")
        parser.add_argument('--gpu_lst', nargs='+', type=int, help="Which gpu to use.")
        parser.add_argument('--LPE', action='store_false', help="Laplacian positional encoding.")
        parser.add_argument('--TE', action='store_false', help="Temporal positional encoding.")
        parser.add_argument('--SPE', action='store_false', help="Shortest path bias encoding.")
        parser.add_argument('--TIE', action='store_false', help="Temporal interval bias encoding.")
        parser.add_argument('--LCA', action='store_false', help="LCA bias encoding.")
        parser.add_argument('--SD_A', action='store_false', help="Self-Distillation attention scores.")
        parser.add_argument('--SD_B', action='store_false', help="Self-Distillation batch relation scores.")
        parser.add_argument('--observation', type=str, default="")
        parser.add_argument('--data_name', type=str, default="weibo")
        return parent_parser


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_dim, ffn_dim):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, ffn_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(ffn_dim, hidden_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, attn_dropout_rate):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attn_dim = attn_dim = hidden_dim // num_heads
        self.scale = attn_dim ** -0.5

        self.linear_Q = nn.Linear(hidden_dim, num_heads * attn_dim)
        self.linear_K = nn.Linear(hidden_dim, num_heads * attn_dim)
        self.linear_V = nn.Linear(hidden_dim, num_heads * attn_dim)

        self.linear_out = nn.Linear(num_heads * attn_dim, hidden_dim)

        self.attn_dropout = nn.Dropout(attn_dropout_rate)

    def forward(self, q, k, v, attn_bias=None, attn_mask=None):
        d_q = d_k = d_v = self.attn_dim  # retaining input.size == output.size
        batch_size = q.size(0)

        # computing x = (Q*k^T)/sqrt(d_q) + attn_bias
        q = self.linear_Q(q).view(batch_size, -1, self.num_heads, d_q)  # (batch_size, seq_len, num_heads, attn_dim)
        k = self.linear_K(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_V(v).view(batch_size, -1, self.num_heads, d_v)

        # q, v: (batch_size, num_heads, seq_len, attn_dim), k: (batch_size, num_heads, attn_dim, seq_len)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2).transpose(2, 3), v.transpose(1, 2)

        x = torch.matmul(q, k) * self.scale
        if attn_bias is not None:
            x = x + attn_bias
        # obtaining attention scores by softmax operation
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        x.masked_fill_(attn_mask, -1e9)
        attention_scores = torch.softmax(x, dim=-1)
        x = self.attn_dropout(attention_scores)
        x = torch.matmul(x, v)

        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.num_heads * self.attn_dim)  # (batch_size, seq_len, num_heads * attn_dim)

        x = self.linear_out(x)  # (batch_size, seq_len, d)
        return x, attention_scores


class GraphEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout_rate, attn_dropout_rate, ffn_dim):
        super(GraphEncoderLayer, self).__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.attention = MultiHeadAttention(hidden_dim, num_heads, attn_dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn_dropout = nn.Dropout(dropout_rate)
        self.ffn = FeedForwardNetwork(hidden_dim, ffn_dim)

    def forward(self, x, attn_bias=None, attn_mask=None):
        y = self.attention_norm(x)
        y, attn = self.attention(y, y, y, attn_bias, attn_mask)
        y = self.attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x, attn


class CTLoss(nn.Module):
    def __init__(self):
        super(CTLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pre, tgt, is_sd=0, sad=None, sbd=None, alpha=0, beta=0, SD_A=False, SD_B=False):
        loss = self.mse_loss(y_pre, tgt)
        if is_sd:
            sad_loss, sbd_loss = 0, 0
            for l in range(1, len(sad)):
                sad_loss += torch.sum((sad[l]-sad[l-1])**2)
                sbd_loss += torch.sum((sbd[l]-sbd[l-1])**2)
            if SD_A:
                loss += alpha * sad_loss
            if SD_B:
                loss += beta * sbd_loss
            return loss
        return loss
