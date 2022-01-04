import torch
from torch_geometric.datasets import DBP15K
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import read_txt_array
from torch_geometric.utils import sort_edge_index
import torch.nn.functional as F
import numpy as np
import argparse
import torch.nn.functional as F
import os.path as osp
import json
from itertools import repeat, product


class MeanEmbedding(object):
    def __call__(self, data):
        data.x1, data.x2 = data.x1.mean(dim=1), data.x2.mean(dim=1)
        return data
class SumEmbedding(object):
    def __call__(self, data):
        data.x1, data.x2 = data.x1.sum(dim=1), data.x2.sum(dim=1)
        return data
class DBP_100k(InMemoryDataset):
    def __init__(self, args, transform = None, pre_transform = None):
        assert args.category in ['WD', 'YG']
        root = './data/DWY100K_1/DWY100K/'
        # self.raw_dir = '/Users/zy/code/dual-consensus/data/DWY100K/'
        if args.category == 'WD':
            self.pair = 'WD'
        elif args.category == 'YG':
            self.pair = 'YG'

        super(DBP_100k, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])




    def process(self):

        emb1_path = osp.join(self.raw_dir, self.pair, 'embedding1')
        emb2_path = osp.join(self.raw_dir, self.pair, 'embedding2')
        g1_path = osp.join(self.raw_dir, self.pair, 'triples_1')

        g2_path = osp.join(self.raw_dir, self.pair, 'triples_2')


        x1, edge_index1, rel1, assoc1 = self.process_graph(
            g1_path, emb1_path)
        x2, edge_index2, rel2, assoc2 = self.process_graph(
            g2_path, emb2_path)

        train_path = osp.join(self.raw_dir, self.pair, 'links')
        train_y = self.process_y(train_path, assoc1, assoc2)

        # test_path = osp.join(self.raw_dir, self.pair, 'test.examples.1000')
        # test_y = self.process_y(test_path, assoc1, assoc2)

        data = Data(x1=x1, edge_index1=edge_index1, rel1=rel1, x2=x2,
                    edge_index2=edge_index2, rel2=rel2, y = train_y)

        torch.save(self.collate([data]), self.processed_paths[0])

    def process_y(self, path, assoc1, assoc2):
        row, col = read_txt_array(path, sep='\t', dtype=torch.long).t()

        return torch.stack([assoc1[row], assoc2[col]], dim=0)

    @property
    def raw_file_names(self):
        return ['WD', 'YG']

    @property
    def processed_file_names(self):
        return '{}.pt'.format(self.pair)

    # def process_graph(self, triple_path, feature_path, embeddings):
    #     g1 = read_txt_array(triple_path, sep=' ', dtype=torch.long)
    #     subj, rel, obj = g1.t()
    #
    #     x_dict = {}
    #     with open(feature_path, 'r', encoding='utf-8') as f:
    #         for line in f:
    #             info = line.strip().split('\t')
    #             info = info if len(info) == 2 else info + ['**UNK**']
    #             seq = info[1].lower().split()
    #             hs = [embeddings.get(w, embeddings['**UNK**']) for w in seq]
    #             x_dict[int(info[0])] = torch.stack(hs, dim=0)
    #
    #     idx = torch.tensor(list(x_dict.keys()))
    #     assoc = torch.full((idx.max().item() + 1,), -1, dtype=torch.long)
    #     assoc[idx] = torch.arange(idx.size(0))
    #
    #     subj, obj = assoc[subj], assoc[obj]
    #     edge_index = torch.stack([subj, obj], dim=0)
    #     edge_index, rel = sort_edge_index(edge_index, rel)  # 某种排序
    #
    #     xs = [None for _ in range(idx.size(0))]
    #     for i in x_dict.keys():
    #         xs[assoc[i]] = x_dict[i]
    #     x = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
    #
    #     return x, edge_index, rel, assoc

    def process_graph(self, triple_path, embeddingpath):
        g1 = read_txt_array(triple_path, sep='\t', dtype=torch.long)
        subj, rel, obj = g1.t()
        x_dict = {}
        with open(embeddingpath, mode = 'r',encoding='utf-8') as f:
            dict = json.load(f)
        for key, em in dict.items():
            x_dict[int(key)] = F.normalize(torch.tensor(em).unsqueeze(0),dim = 1)

        idx = torch.tensor(list(x_dict.keys()))
        assoc = torch.full((idx.max().item() + 1,), -1, dtype=torch.long)
        assoc[idx] = torch.arange(idx.size(0))

        subj, obj = assoc[subj], assoc[obj]
        edge_index = torch.stack([subj, obj], dim=0)
        edge_index, rel = sort_edge_index(edge_index, rel)  # 某种排序

        xs = [None for _ in range(idx.size(0))]
        for i in x_dict.keys():
            xs[assoc[i]] = x_dict[i]
        x = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)

        return x, edge_index, rel, assoc



class DBP_100k_new():
    def __init__(self,args):
        data = DBP_100k(args, transform=MeanEmbedding(), pre_transform=None)[0]
        # data = DBP15K(args.data_path, args.category, transform=SumEmbedding())[0]
        self.edge_index1 = data.edge_index1.to(args.device)
        self.edge_index2 = data.edge_index2.to(args.device)
        self.rel1 = data.rel1
        self.rel2 = data.rel2
        self.H1, self.G1, self.rel1, self.dual_adj1 = self.process_relation(self.rel1, self.edge_index1)
        self.H2, self.G2, self.rel2, self.dual_adj2 = self.process_relation(self.rel2, self.edge_index2)
        self.H1 = torch.tensor(self.H1).to_sparse().float().to(args.device)
        self.G1 = torch.tensor(self.G1).to_sparse().float().to(args.device)
        self.H2 = torch.tensor(self.H2).to_sparse().float().to(args.device)
        self.G2 = torch.tensor(self.G2).to_sparse().float().to(args.device)
        self.dual_adj1 = torch.tensor(self.dual_adj1).float().to_sparse().to(args.device)
        self.dual_adj2 = torch.tensor(self.dual_adj2).float().to_sparse().to(args.device)
        # self.y = data.y
        # self.test_y = data.test_y
        self.data = np.transpose(data.y.numpy())
        np.random.shuffle(self.data)
        self.data = torch.tensor(np.transpose(self.data))
        length = self.data.size(1)
        num1 = int(length*args.p)
        self.train_y = self.data[:, 0:num1].to(args.device)
        self.test_y = self.data[:, num1:].to(args.device)
        self.x1 = data.x1.to(args.device)
        self.x1 = F.normalize(self.x1, p=2, dim=1)
        self.x2 = data.x2.to(args.device)
        self.x2 = F.normalize(self.x2, p=2, dim=1)


    def process_relation(self, rel, edge_index):
        num_entities = edge_index.max().item() + 1
        newrel = rel - rel.min()
        num_relations = newrel.max().item() + 1
        H1 = np.zeros((num_entities, num_relations))
        G1 = np.zeros((num_entities, num_relations))
        head = {}
        tail = {}
        cnt = {}
        for k in range(len(newrel)):
            H1[edge_index[0, k], newrel[k]] = 1
            G1[edge_index[1, k], newrel[k]] = 1
            if newrel[k].item() not in cnt:
                cnt[newrel[k].item()] = 1
                head[newrel[k].item()] = set([edge_index[0, k].item()])
                tail[newrel[k].item()] = set([edge_index[1, k].item()])
            else:
                cnt[newrel[k].item()] += 1
                head[newrel[k].item()].add(edge_index[0, k].item())
                tail[newrel[k].item()].add(edge_index[1, k].item())
        dual_adj = np.zeros((num_relations, num_relations))
        newrel = newrel.numpy()
        for i in range(newrel.max() + 1):
            for j in range(i + 1):
                a1 = len(head[i] & head[j]) / len(head[i] | head[j])
                a2 = len(tail[i] & tail[j]) / len(tail[i] | tail[j])
                dual_adj[i, j] = (a1 + a2) / 2
        dual_adj += dual_adj.T - np.diag(dual_adj.diagonal())
        return H1, G1, newrel, dual_adj

class DBP15K_new():
    def __init__(self, args):
        data = DBP15K(args.data_path, args.category, transform=SumEmbedding())[0]
        np.random.seed(args.seed)
        self.edge_index1 = data.edge_index1.to(args.device)
        self.edge_index2 = data.edge_index2.to(args.device)
        self.rel1 = data.rel1
        self.rel2 = data.rel2
        self.H1, self.G1, self.rel1, self.dual_adj1 = self.process_relation(self.rel1, self.edge_index1)
        self.H2, self.G2, self.rel2, self.dual_adj2 = self.process_relation(self.rel2, self.edge_index2)
        self.H1 = torch.tensor(self.H1).to_sparse().float().to(args.device)
        self.G1 = torch.tensor(self.G1).to_sparse().float().to(args.device)
        self.H2 = torch.tensor(self.H2).to_sparse().float().to(args.device)
        self.G2 = torch.tensor(self.G2).to_sparse().float().to(args.device)
        self.dual_adj1 = torch.tensor(self.dual_adj1).float().to_sparse().to(args.device)
        self.dual_adj2 = torch.tensor(self.dual_adj2).float().to_sparse().to(args.device)
        self.train_y = data.train_y
        self.test_y = data.test_y
        self.data = np.transpose(torch.cat((self.train_y, self.test_y), 1).numpy())
        np.random.shuffle(self.data)
        self.data = torch.tensor(np.transpose(self.data))
        length = self.data.size(1)
        num1 = int(length*args.p)
        self.train_y = self.data[:, 0:num1].to(args.device)
        self.test_y = self.data[:, num1:].to(args.device)
        self.x1 = data.x1.to(args.device)
        # self.x1 = F.normalize(self.x1, p=2, dim=1)
        self.x2 = data.x2.to(args.device)
        # self.x2 = F.normalize(self.x2, p=2, dim=1)


    def process_relation(self, rel, edge_index):
        num_entities = edge_index.max().item() + 1
        newrel = rel - rel.min()
        num_relations = newrel.max().item()+ 1
        H1 = np.zeros((num_entities, num_relations))
        G1 = np.zeros((num_entities, num_relations))
        head = {}
        tail = {}
        cnt = {}
        for k in range(len(newrel)):
            H1[edge_index[0, k], newrel[k]] = 1
            G1[edge_index[1, k], newrel[k]] = 1
            if newrel[k].item() not in cnt:
                cnt[newrel[k].item()] = 1
                head[newrel[k].item()] = set([edge_index[0, k].item()])
                tail[newrel[k].item()] = set([edge_index[1, k].item()])
            else:
                cnt[newrel[k].item()] += 1
                head[newrel[k].item()].add(edge_index[0, k].item())
                tail[newrel[k].item()].add(edge_index[1, k].item())
        dual_adj = np.zeros((num_relations, num_relations))
        newrel = newrel.numpy()
        for i in range(newrel.max()+1):
            for j in range(i+1):
                a1 = len(head[i] & head[j])/len(head[i]| head[j])
                a2 = len(tail[i] & tail[j])/len(tail[i]| tail[j])
                dual_adj[i, j] = (a1 + a2)/2
        dual_adj += dual_adj.T - np.diag(dual_adj.diagonal())
        return H1, G1, newrel, dual_adj




if __name__ == '__main__':
    def parse_args(args=None):
        parser = argparse.ArgumentParser(
            description='Training and Testing dual-consensus Models',
            usage='main.py [<args>] [-h | --help]'
        )

        parser.add_argument('--category', default='WD', type=str)
        parser.add_argument('--dim', type=int, default=256)
        parser.add_argument('--rnd_dim', type=int, default=32)
        parser.add_argument('--num_layers', type=int, default=3)
        parser.add_argument('--num_steps', type=int, default=10)
        parser.add_argument('--sinknum', type=int, default=10)
        # parser.add_argument('--WD', type=str, default='')

        parser.add_argument('--k', type=int, default=10)
        parser.add_argument('--seed', type=int, default=123)
        parser.add_argument('--p', type=float, default=0.3)

        parser.add_argument('--cuda', action='store_true', default=False, help='use GPU')
        parser.add_argument('--device', type=str, default='cpu', help='')
        parser.add_argument('-save', '--save_path', default='./save/wn18_match', type=str)
        parser.add_argument('-init', '--init_checkpoint', default=None, type=str)

        parser.add_argument('--do_train', action='store_true', default=False)
        parser.add_argument('--do_valid', action='store_true', default=False)
        parser.add_argument('--do_test', action='store_true', default=False)

        parser.add_argument('--evaluate_train', action='store_true', default=False, help='Evaluate on training data')
        parser.add_argument('--data_path', type=str, default='./data/DBP15K')

        parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
        parser.add_argument('-d', '--hidden_dim', default=300, type=int)
        parser.add_argument('-g', '--gamma', default=12.0, type=float)
        parser.add_argument('-b', '--batch_size', default=300, type=int)
        parser.add_argument('-r', '--regularization', default=0, type=float)
        parser.add_argument('--test_batch_size', default=16, type=int, help='valid/test batch size')
        parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
        parser.add_argument('-cpu', '--cpu_num', default=10, type=int)

        parser.add_argument('--max_steps', default=300000, type=int)
        parser.add_argument('--change_steps', default=100000, type=int)
        parser.add_argument('--warm_up_steps', default=None, type=int)
        parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
        parser.add_argument('--valid_steps', default=30000, type=int)
        parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
        parser.add_argument('--test_log_steps', default=10000, type=int, help='valid/test log every xx steps')
        return parser.parse_args(args)

    args = parse_args()
    # data = DBP_100k(args, transform=SumEmbedding(), pre_transform =None)[0]
    pf = DBP_100k_new(args)
    p= 2
