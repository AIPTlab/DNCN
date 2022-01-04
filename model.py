import torch
from torch.nn import Linear as Lin, BatchNorm1d as BN
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv
from data import DBP15K_new
from torch_scatter import  scatter_add
import argparse
import numpy as np
import time
import torch.nn.functional as F
class RelConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(RelConv, self).__init__(aggr='mean')

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin1 = Lin(in_channels, out_channels, bias=False)
        self.lin2 = Lin(in_channels, out_channels, bias=False)
        self.root = Lin(in_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.root.reset_parameters()

    def forward(self, x, edge_index):
        """"""
        self.flow = 'source_to_target'
        out1 = self.propagate(edge_index, x=self.lin1(x))
        self.flow = 'target_to_source'
        out2 = self.propagate(edge_index, x=self.lin2(x))
        return self.root(x) + out1 + out2

    def message(self, x_j):
        return x_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class RelCNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, batch_norm=False,
                 cat=True, lin=True, dropout=0.0):
        super(RelCNN, self).__init__()

        self.in_channels = in_channels
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.cat = cat
        self.lin = lin
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(RelConv(in_channels, out_channels))
            self.batch_norms.append(BN(out_channels))
            in_channels = out_channels

        if self.cat:
            in_channels = self.in_channels + num_layers * out_channels
        else:
            in_channels = out_channels

        if self.lin:
            self.out_channels = out_channels
            self.final = Lin(in_channels, out_channels)
        else:
            self.out_channels = in_channels

        self.reset_parameters()

    def reset_parameters(self):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            conv.reset_parameters()
            batch_norm.reset_parameters()
        if self.lin:
            self.final.reset_parameters()

    def forward(self, x, edge_index, *args):
        """"""
        xs = [x]

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = conv(xs[-1], edge_index)
            x = batch_norm(F.relu(x)) if self.batch_norm else F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)

        x = torch.cat(xs, dim=-1) if self.cat else xs[-1]
        x = self.final(x) if self.lin else x
        return x

    def __repr__(self):
        return ('{}({}, {}, num_layers={}, batch_norm={}, cat={}, lin={}, '
                'dropout={})').format(self.__class__.__name__,
                                      self.in_channels, self.out_channels,
                                      self.num_layers, self.batch_norm,
                                      self.cat, self.lin, self.dropout)


class Sinkhorn(torch.nn.Module):
    """
    BiStochastic Layer turns the input matrix into a bi-stochastic matrix.
    Parameter: maximum iterations max_iter
               a small number for numerical stability epsilon
    Input: input matrix s
    Output: bi-stochastic matrix s
    """
    def __init__(self, max_iter=10, epsilon=1e-4):
        super(Sinkhorn, self).__init__()
        self.max_iter = max_iter
        self.epsilon = epsilon

    def forward(self, s, nrows=None, ncols=None, exp=False, exp_alpha=2, dummy_row=False, dtype=torch.float32):
        batch_size = s.shape[0]

        if dummy_row:
            dummy_shape = list(s.shape)
            dummy_shape[1] = s.shape[2] - s.shape[1] # row<col
            s = torch.cat((s, torch.full(dummy_shape, 0.).to(s.device)), dim=1)
            new_nrows = ncols
            for b in range(batch_size):
                s[b, nrows[b]:new_nrows[b], :ncols[b]] = self.epsilon
            nrows = new_nrows

        row_norm_ones = torch.zeros(batch_size, s.shape[1], s.shape[1], device=s.device)  # size: row x row
        col_norm_ones = torch.zeros(batch_size, s.shape[2], s.shape[2], device=s.device)  # size: col x col
        for b in range(batch_size):
            row_slice = slice(0, nrows[b] if nrows is not None else s.shape[2])
            col_slice = slice(0, ncols[b] if ncols is not None else s.shape[1])
            row_norm_ones[b, row_slice, row_slice] = 1
            col_norm_ones[b, col_slice, col_slice] = 1

        # for Sinkhorn stacked on last dimension
        if len(s.shape) == 4:
            row_norm_ones = row_norm_ones.unsqueeze(-1)
            col_norm_ones = col_norm_ones.unsqueeze(-1)

        s += self.epsilon
        if exp:
            s = torch.exp(exp_alpha * s)
        s = s.squeeze(0)
        for i in range(self.max_iter):
            # start = time.time()
            if i % 2 == 1:
                # sum = torch.sum(torch.mul(s.unsqueeze(3), col_norm_ones.unsqueeze(1)), dim=2) # 1 3 3 1 * 1 1 3 3 = 1 3 3 3
                row_sum = 1 / (s.sum(dim = 1).view(s.size(0),1).repeat(1, s.size(1)))
                s = s * row_sum
            else:
                # row norm
                col_sum = 1 / (s.sum(dim = 0).repeat(s.size(0), 1))
                s = s * col_sum
                # sum = torch.sum(torch.mul(row_norm_ones.unsqueeze(3), s.unsqueeze(1)), dim=2) # 1 3 3 1 * 1 1 3 3  = 1 3 3 3 sum 1 3 3
            # end = time.time()
            # print('sinkhorn one Running time: %s Seconds' % (end - start))
            # tmp = torch.zeros_like(s)
            # for b in range(batch_size):
            #     row_slice = slice(0, nrows[b] if nrows is not None else s.shape[2])
            #     col_slice = slice(0, ncols[b] if ncols is not None else s.shape[1])
            #     tmp[b, row_slice, col_slice] = 1 / sum[b, row_slice, col_slice]
            # s = s * tmp

        if dummy_row and dummy_shape[1] > 0:
            s = s[:-dummy_shape[1],:]

        return s


class DualConsensusNet(torch.nn.Module):
    def __init__(self, args):
        super(DualConsensusNet, self).__init__()
        self.embedding_shape = args.embedding_shape
        self.PrimaryConv = RelCNN(args.inshape, args.embedding_shape, args.num_layers, batch_norm=False,
               cat=True, lin=True, dropout=0.5)

        self.PrimSink = Sinkhorn(args.sinknum, 1e-10) #epsilon = 1e-10
        self.DualConv = GCNConv(args.embedding_shape * 2, args.embedding_shape, add_self_loops = False)
        self.DualSink = Sinkhorn(args.sinknum, 1e-10)

        self.negative_number = args.negative_number
        self.color_size = args.color_size
        self.SConv = RelCNN(self.color_size, self.color_size, args.num_layers, batch_norm=False,
                            cat=True, lin=True, dropout=0.5)
        self.mlp = Seq(
            Lin(args.color_size, args.color_size),
            ReLU(),
            Lin(args.color_size, 1),
        )
        self.begin_steps = args.begin_steps
        self.train_steps = 0


    def forward(self, x_s, x_t, edges, edget, Hs, Gs, Ht, Gt, dedges, dws, dedget, dwt, y = None):
        h_s = self.PrimaryConv(x_s, edges) # (ns, args.embedding_shape)
        h_t = self.PrimaryConv(x_t, edget)# (nt, args.embedding_shape)
        # h_s = F.normalize(h_s, p=2, dim=1)
        # h_t = F.normalize(h_t, p=2, dim=1)
        #primary matching
        S_sparse_0, S_hat = self.primaryMatch(h_s, h_t, y)
        S_1 = None
        if self.train_steps >= self.begin_steps:
            #dual graph constrution
            dh_s0 = self.construct_dual_embedding(h_s, Hs, Gs) # nr 2d
            dh_t0 = self.construct_dual_embedding(h_t, Ht, Gt)
            #dual convolution
            # dh_s = self.DualConv(dh_s0, dedges, dws)
            # dh_t = self.DualConv(dh_t0, dedget, dwt)
            dh_s = F.normalize(self.DualConv(dh_s0, dedges, dws), p = 2, dim = 1)
            dh_t = F.normalize(self.DualConv(dh_t0, dedget, dwt), p = 2 ,dim = 1)
            #dual matching
            edge_match = torch.mm(dh_s, dh_t.transpose(-1, -2))
            row, col = edge_match.size()
            edge_match = edge_match.unsqueeze(0)
            if row <= col:
                edge_match = self.DualSink(edge_match, nrows =[row], ncols=[col], exp_alpha = 20, dummy_row=True, exp=True)
                edge_match = edge_match.squeeze(0)
            else:
                edge_match = edge_match.transpose(-1, -2)
                edge_match = self.DualSink(edge_match, nrows=[col], ncols=[row], exp_alpha = 20,  dummy_row=True, exp=True)
                edge_match = edge_match.squeeze(0).transpose(-1, -2) # ns * nt
            S_idx = S_sparse_0.__idx__
            for _ in range(10):
                S = S_hat.softmax(dim=-1) # ns 20

                # edge coloring
                # rs = torch.tensor(np.random.randint(0, 2, (dh_s0.size(0), self.color_size)), dtype=dh_s0.dtype,
                #                   device=dh_s0.device)
                rs = torch.randn((dh_s0.size(0), self.color_size), dtype = dh_s0.dtype, device = dh_s0.device) # nr s
                rt = torch.mm(edge_match.transpose(0, 1), rs)  # nt s
                colorEs = self.edge_color2node(rs, Hs, Gs)
                colorEt = self.edge_color2node(rt, Ht, Gt)
                o_s = self.SConv(colorEs, edges)
                o_t = self.SConv(colorEt, edget)
                #node coloring

                es = torch.randn((h_s.size(0), self.color_size), dtype = h_s.dtype, device = h_s.device)
                tmp_t = es.view(h_s.size(0), 1, self.color_size) * S.view( h_s.size(0), S.size(1), 1)
                tmp_t = tmp_t.view( h_s.size(0) * S.size(1), self.color_size)
                idx = S_idx.view( h_s.size(0) * S.size(1), 1)
                et = scatter_add(tmp_t, idx, dim=0, dim_size=h_t.size(0))
                o_s1 = self.SConv(es, edges)
                o_t1 = self.SConv(et, edget)
                D = self.refine(o_s, o_t, S_idx)
                D1 = self.refine(o_s1, o_t1, S_idx)
                S_hat = S_hat + self.mlp(D).squeeze(-1) + self.mlp(D1).squeeze(-1)
                # S_hat = S_hat + self.mlp(D1).squeeze(-1)
                # S_hat = S_hat + self.mlp(D).squeeze(-1)


            S_d = S_hat.softmax(dim=-1)
            row = torch.arange(x_s.size(0), device=S_idx.device)
            row = row.view(-1, 1).repeat(1, S_idx.size(1))
            idx = torch.stack([row.view(-1), S_idx.view(-1)], dim=0)
            size = torch.Size([x_s.size(0),x_t.size(0)])

            S_1 = torch.sparse_coo_tensor(
                idx, S_d.view(-1), size, requires_grad=S_d.requires_grad)
            S_1.__idx__ = S_idx
            S_1.__val__ = S_d
            # S_1 = self.primaryMatch(colorEs, colorEt, y)

        return S_sparse_0, S_1

    def refine(self,o_s, o_t,S_idx):
        o_s = o_s.view(o_s.size(0), 1, o_s.size(1)).expand(-1, S_idx.size(1), -1)
        idx = S_idx.view(o_s.size(0) * S_idx.size(1), 1).expand(-1, self.color_size)
        tmp_t = torch.gather(o_t.view(o_t.size(0), self.color_size), -2, idx)
        D = o_s - tmp_t.view(o_s.size(0), S_idx.size(1), self.color_size)
        return D
    def acc(self, S, y):
        r"""Computes the accuracy of correspondence predictions.

        Args:
            S (Tensor): Sparse or dense correspondence matrix of shape
                :obj:`[ num_nodes, num_nodes]`.
            y (LongTensor): Ground-truth matchings of shape
                :obj:`[2, num_ground_truths]`.
            reduction (string, optional): Specifies the reduction to apply to
                the output: :obj:`'mean'|'sum'`. (default: :obj:`'mean'`)
        """

        if not S.is_sparse:
            pred = S[y[0]].argmax(dim=-1)
        else:
            assert S.__idx__ is not None and S.__val__ is not None
            pred = S.__idx__[y[0], S.__val__[y[0]].argmax(dim=-1)]

        correct = (pred == y[1]).sum().item()
        t = (y[1].view(-1, 1) == S.__idx__[y[0],:]).nonzero()
        rank = t[:, 1].view(1, -1) + 1
        return correct / y.size(1), torch.mean(1.0/rank)

    def MRR(self, S, y):
        t = (y[1].view(-1, 1) == S.__idx__[y[0],:]).nonzero()
        rank = t[:, 1].view(1, -1) + 1
        return torch.mean(1.0/rank)

    def hits_at_k(self, k, S, y):
        r"""Computes the hits@k of correspondence predictions.

        Args:
            k (int): The :math:`\mathrm{top}_k` predictions to consider.
            S (Tensor): Sparse or dense correspondence matrix of shape
                :obj:`[batch_size * num_nodes, num_nodes]`.
            y (LongTensor): Ground-truth matchings of shape
                :obj:`[2, num_ground_truths]`.
            reduction (string, optional): Specifies the reduction to apply to
                the output: :obj:`'mean'|'sum'`. (default: :obj:`'mean'`)
        """

        if not S.is_sparse:
            pred = S[y[0]].argsort(dim=-1, descending=True)[:, :k]
        else:
            assert S.__idx__ is not None and S.__val__ is not None
            perm = S.__val__[y[0]].argsort(dim=-1, descending=True)[:, :k]
            pred = torch.gather(S.__idx__[y[0]], -1, perm)

        correct = (pred == y[1].view(-1, 1)).sum().item()
        return correct / y.size(1)

    def loss(self, S, y):
        mask = S.__idx__[y[0]] == y[1].view(-1, 1)
        positive_val = S.__val__[[y[0]]][mask] #  取train的那些，同时mask锁定预测对的那些value,令其足够的大 考虑需不需要加入反例
        negative_val = S.__val__[[y[0]]][~mask]
        postive_loss = -torch.log(positive_val + 1e-8)
        negative_loss = -torch.log(negative_val + 1e-8)
        # postive_loss = -F.logsigmoid(positive_val)
        # negative_loss = -F.logsigmoid(-negative_val)
        return torch.mean(postive_loss), torch.mean(negative_loss)

    @staticmethod
    def train_step(model, optimizer, data, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''
        model.train()
        optimizer.zero_grad()
        x_s = data.x1
        x_t = data.x2
        edges = data.edge_index1
        edget = data.edge_index2
        Hs = data.H1
        Ht = data.H2
        Gs = data.G1
        Gt = data.G2
        dedges = data.dual_adj1.indices()
        dws = data.dual_adj1.values()
        dedget = data.dual_adj2.indices()
        dwt = data.dual_adj2.values()
        t1  = time.time()
        S_0, S_1 = model(x_s, x_t, edges, edget, Hs, Gs, Ht, Gt, dedges, dws, dedget, dwt, data.train_y)
        model.train_steps += 1
        if S_1 is not None:
            posi_score, nega_score = model.loss(S_1, data.train_y)
            # posi_score2, nega_score2 = model.loss(S_1, data.train_y)
            # posi_score = (posi_score1 + args.beta1 * posi_score2) / (1 + args.beta1)
            # nega_score = (nega_score1 + args.beta1 * nega_score2) / (1 + args.beta1)
        else:
            posi_score, nega_score = model.loss(S_0, data.train_y)

        # loss = (posi_score - args.beta2 * nega_score)/ (1 + args.beta2)
        loss = posi_score
        # print(loss)
        # print(model.train_steps)
        loss.backward()
        optimizer.step()
        t2 = time.time()
        print('training one epoch time is ', t2 - t1)
        log = {
            'positive_sample_loss': posi_score.item(),
            'negative_sample_loss': nega_score.item(),
            'loss': loss.item()}
        return log

    @staticmethod
    def test_step(model, data,  args, set = 'test'):
        '''
        Evaluate the model on test or valid datasets
        '''
        model.eval()
        # Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
        # Prepare dataloader for evaluation
        x_s = data.x1
        x_t = data.x2
        edges = data.edge_index1
        edget = data.edge_index2
        Hs = data.H1
        Ht = data.H2
        Gs = data.G1
        Gt = data.G2
        dedges = data.dual_adj1.indices()
        dws = data.dual_adj1.values()
        dedget = data.dual_adj2.indices()
        dwt = data.dual_adj2.values()
        t1 = time.time()
        S_0, S_1 = model(x_s, x_t, edges, edget, Hs, Gs, Ht, Gt, dedges, dws, dedget, dwt)
        t2 = time.time()
        print('inference time is ', t2 - t1)
        if S_1 is not None:
            S = S_1
        else:
            S = S_0
        if set =='test':
            hits1,mrr = model.acc(S, data.test_y)
            hits3 = model.hits_at_k(3, S, data.test_y)
            hits10 = model.hits_at_k(10, S, data.test_y)
            mrr = model.MRR( S, data.test_y)
        elif set == 'train':
            hits1, mrr = model.acc(S, data.train_y)
            hits3 = model.hits_at_k(3, S, data.train_y)
            hits10 = model.hits_at_k(10, S, data.train_y)
            # mrr = model.MRR(S, data.train_y)
        logs = {
            'hits1': hits1,
            'hits3': hits3,
            'hits10': hits10,
            'mrr':mrr }
        return logs

    def __top_k__(self, x_s, x_t):
        x_t = x_t.transpose(-1, -2)  # [..., d, n_t]
        S_ij = x_s @ x_t
        return S_ij.topk(self.negative_number, dim=1)[1]

    def primaryMatch(self, x_s, x_t, y = None):
        S_idx = self.__top_k__(x_s, x_t)
        N_s = x_s.size(0)
        N_t = x_t.size(0)
        if y is not None:
            rnd_size = (N_s, self.negative_number)
            S_rnd_idx = torch.randint(N_t, rnd_size, dtype=torch.long, device=S_idx.device)
            S_idx = torch.cat([S_idx, S_rnd_idx], dim=-1)
            S_idx, mask = self.__include_gt__(S_idx, y) #如果该行所有的都预测不对，在该行最后一列添加label
        k = S_idx.size(-1)
        C_out = x_s.size(-1)
        tmp_s = x_s.view(N_s, 1, C_out) #[19780, 1, 300])
        idx = S_idx.view(N_s * k, 1).expand(-1, C_out)  # 找到20个所有的embedding
        tmp_t = torch.gather(x_t.view(N_t, C_out), -2, idx)
        S_hat = (tmp_s * tmp_t.view( N_s, k, C_out)).sum(dim=-1)  # 和所有候选的做内积
        S_0 = S_hat.softmax(dim=-1)

        row = torch.arange(x_s.size(0), device=S_idx.device)
        row = row.view(-1, 1).repeat(1, k)
        idx = torch.stack([row.view(-1), S_idx.view(-1)], dim=0)
        size = torch.Size([x_s.size(0), N_t])

        S_sparse_0 = torch.sparse_coo_tensor(
            idx, S_0.view(-1), size, requires_grad=S_0.requires_grad)
        S_sparse_0.__idx__ = S_idx
        S_sparse_0.__val__ = S_0
        return S_sparse_0,S_hat

    def edge_color2node(self, r, H, G):
        #H: ns nr r: nr d , G: ns nr
        cs = torch.sparse.mm(H, r)
        ct = torch.sparse.mm(G, r)
        # torch.cat((cs, ct), dim=1)
        return cs + ct

    def __include_gt__(self, S_idx, y):
        (row, col), (N_s, k) = y, S_idx.size()
        
        gt_mask = (S_idx[row] != col.view(-1, 1)).all(dim=-1)
        sparse_mask = gt_mask.new_zeros((N_s,))
        sparse_mask[row] = gt_mask
        label = torch.full((N_s, ), -1, device = S_idx.device)
        label[row] = col
        last_entry = torch.zeros(k, dtype=torch.bool, device=gt_mask.device)
        last_entry[-1] = 1  # [20] 最后一个位置是1
        dense_mask = sparse_mask.view(N_s, 1) * last_entry.view(1, k)  # 最后一列没预测出来的行为True
        return S_idx.masked_scatter(dense_mask, label[sparse_mask]), gt_mask
        #dual consensus


    def construct_dual_embedding(self, h, H, G):
        d_h = torch.sparse.mm(torch.transpose(H, 0, 1), h) / \
                 torch.sparse.sum(H, 0).to_dense().view(H.size(1), -1)  #(nr, ns) (ns d) -> nr d
        d_t = torch.sparse.mm(torch.transpose(G, 0, 1), h) / \
                 torch.sparse.sum(G, 0).to_dense().view(G.size(1), -1) # (nr d)
        # (nr)
        return torch.cat((d_h, d_t), dim = 1)# (nr, 2d)


# if __name__ == '__main__':
#     def parse_args(args=None):
#         parser = argparse.ArgumentParser(
#             description='Training and Testing dual-consensus Models',
#             usage='main.py [<args>] [-h | --help]'
#         )
#
#         parser.add_argument('--category', default='en_ja', type=str)
#         parser.add_argument('--dim', type=int, default=256)
#         parser.add_argument('--rnd_dim', type=int, default=32)
#         parser.add_argument('--num_layers', type=int, default=3)
#         parser.add_argument('--inshape', type=int, default=300)
#         parser.add_argument('--embedding_shape', type=int, default=300)
#         parser.add_argument('--sinknum', type=int, default=10)
#         parser.add_argument('--color_size', type=int, default=30)
#         parser.add_argument('--begin_steps', type=int, default = 1000)
#
#         parser.add_argument('--beta1', type=float, default=1.0)
#         parser.add_argument('--beta2', type=float, default=1.0)
#
#         parser.add_argument('--num_steps', type=int, default=10)
#         parser.add_argument('--k', type=int, default=10)
#         parser.add_argument('--seed', type=int, default=123)
#         parser.add_argument('--p', type=float, default=0.3)
#         parser.add_argument('--negative_number', type=int, default=10)
#         #negative_number
#
#         parser.add_argument('--cuda', action='store_true', default=False, help='use GPU')
#         parser.add_argument('--device', type=str, default='cpu', help='')
#         parser.add_argument('-save', '--save_path', default='./save/wn18_match', type=str)
#         parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
#
#         parser.add_argument('--do_train', action='store_true', default=False)
#         parser.add_argument('--do_valid', action='store_true', default=False)
#         parser.add_argument('--do_test', action='store_true', default=False)
#
#         parser.add_argument('--evaluate_train', action='store_true', default=False, help='Evaluate on training data')
#         parser.add_argument('--data_path', type=str, default='./data/DBP15K')
#
#         parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
#         parser.add_argument('-d', '--hidden_dim', default=300, type=int)
#         parser.add_argument('-g', '--gamma', default=12.0, type=float)
#         parser.add_argument('-b', '--batch_size', default=300, type=int)
#         parser.add_argument('-r', '--regularization', default=0, type=float)
#         parser.add_argument('--test_batch_size', default=16, type=int, help='valid/test batch size')
#         parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
#         parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
#
#         parser.add_argument('--max_steps', default=300000, type=int)
#         parser.add_argument('--change_steps', default=100000, type=int)
#         parser.add_argument('--warm_up_steps', default=None, type=int)
#         parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
#         parser.add_argument('--valid_steps', default=30000, type=int)
#         parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
#         parser.add_argument('--test_log_steps', default=10000, type=int, help='valid/test log every xx steps')
#         return parser.parse_args(args)
#
#     args = parse_args()
#     data = DBP15K_new(args)
#     model = DualConsensusNet(args)
#     x_s = data.x1
#     x_t = data.x2
#     edges = data.edge_index1
#     edget = data.edge_index2
#     Hs = data.H1
#     Ht = data.H2
#     Gs = data.G1
#     Gt = data.G2
#     dedges = data.dual_adj1.indices()
#     dws = data.dual_adj1.values()
#     dedget =data.dual_adj2.indices()
#     dwt = data.dual_adj2.values()
#
#     gg = model(x_s, x_t, edges, edget, Hs, Gs, Ht, Gt, dedges, dws, dedget, dwt, data.train_y)
#
#
#     p = 1
















