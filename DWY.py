import torch
from torch_geometric.datasets import DBP15K
import numpy as np
import argparse
import torch.nn.functional as F
import os



def preprocess(root, root1, num):
    entities2id = {}
    eid2name = {}
    relation2id = {}
    rid2name = {}
    tri = []
    with open(os.path.join(root, 'entity_local_name_'+str(num)), 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = line.split()
            entity = data[0]
            name = [' '+i for i in data[1:]]
            name = ''.join(name)
            if entity not in entities2id:
                entities2id[entity] = i
                eid2name[i] = name
    r2name = {}
    with open(os.path.join(root, 'predicate_local_name_'+str(num)), 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = line.split()
            relation = data[0]
            name = [' '+i for i in data[1:]]
            name = ''.join(name)
            if relation not in relation2id:
                r2name[relation] = name
    with open(os.path.join(root, 'rel_triples_' + str(num)), 'r', encoding='utf-8') as f:
        tp = 0
        for i, line in enumerate(f):
            data = line.split()
            if data[1] not in relation2id:
                relation2id[data[1]] = tp
                tp = tp +1

    with open(os.path.join(root, 'rel_triples_' + str(num)), 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = line.split()
            tri.append([entities2id[data[0]], relation2id[data[1]], entities2id[data[2]]])

    for re, id in relation2id.items():
        rid2name[id] = r2name[re]
    #save

    with open(os.path.join(root1, 'ent_ids_' + str(num)), 'w', encoding='utf-8') as f:
        for entity, number in entities2id.items():
            f.write(str(number)+'\t'+entity + '\n')

    with open(os.path.join(root1, 'relation_ids_' + str(num)), 'w', encoding='utf-8') as f:
        for relation, number in relation2id.items():
            f.write(str(number)+'\t'+relation + '\n')

    with open(os.path.join(root1, 'id_features_' + str(num)), 'w', encoding='utf-8') as f:
        for id, name in eid2name.items():
            f.write(str(id)+'\t'+ name + '\n')

    with open(os.path.join(root1, 'reid_features_' + str(num)), 'w', encoding='utf-8') as f:
        for id, name in rid2name.items():
            f.write(str(id)+'\t'+name + '\n')

    with open(os.path.join(root1, 'triples_' + str(num)), 'w', encoding='utf-8') as f:
        for l in tri:
            f.write(str(l[0])+' '+str(l[1]) + ' ' + str(l[2]) +'\n')
    return entities2id , eid2name , relation2id , rid2name

root = 'data/DWY100K/DBP_YG_100K'
root1 = 'data/DWY100K/raw/YG'
entities2id1, eid2name1, relation2id1, rid2name1 = preprocess(root, root1, 1)
entities2id2, eid2name2, relation2id2, rid2name2 = preprocess(root, root1, 2)

links = []
with open(os.path.join(root, 'ent_links'), 'r', encoding='utf-8') as f:
    for line in f:
        entity = line.split()
        e1 = entity[0]
        e2 = entity[1]
        try:
            links.append([entities2id1[e1], entities2id2[e2]])
        except:
            p = 1

with open(os.path.join(root1, 'links'),'w') as f:
    for data in links:
        f.write(str(data[0]) + ' ' + str(data[1]) +'\n')

p = 1