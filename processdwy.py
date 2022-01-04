
import json
import torch
from torch_geometric.io import read_txt_array
root = r'F:\zhangyuxuan\DualConsensusNet\data\DWY100K_1\DWY100k\dbp_yg'
feature_path = root + r'\vectorList.json'
id_path1 = root + r'\ent_ids_1'
id_path2 = root + r'\ent_ids_2'
tri1 = root + r'\triples_1'
tri2 = root + r'\triples_2'
link = root + r'\ref_ent_ids'

newroot  = r'F:\zhangyuxuan\DualConsensusNet\data\DWY100K_1\DWY100k\raw\YG'
ntri1 = newroot+ r'\triples_1'
ntri2 = newroot+ r'\triples_2'
nid_path1  = newroot + r'\ent_ids_1'
nid_path2 = newroot + r'\ent_ids_2'
nlink =  newroot + r'\links'
nembedding1  = newroot + r'\embedding1'
nembedding2 = newroot + r'\embedding2'

id2num1 = {}
num2id1 = {}
entity1 = []
entity2 = []
with open(id_path1, encoding='utf-8') as f:
    i = 0
    for line in f:
        th = line[:-1].split('\t')
        id2num1[i] = int(th[0])
        entity1.append(th[1])
        i = i+1

for id, num in id2num1.items():
    num2id1[num] = id

id2num2 = {}
num2id2 = {}
with open(id_path2, encoding='utf-8') as f:
    i = 0
    for line in f:
        th = line[:-1].split('\t')
        id2num2[i] = int(th[0])
        entity2.append(th[1])
        i = i+1
for id, num in id2num2.items():
    num2id2[num] = id

triples1 = []
triples2 = []
id2rel1 = {}
rel2id1 = {}
rel2id2 = {}
id2rel1 = {}
id2rel2 = {}
relations1 = []
relations2  = []
links = []

with open(tri1, encoding='utf-8') as f:
    for line in f:
        th = line[:-1].split('\t')
        relations1.append(int(th[1]))
relations1  = list(set(relations1))

for i, rel in enumerate(relations1):
    id2rel1[i]  = rel
    rel2id1[rel]  = i

with open(tri1, encoding='utf-8') as f:
    for line in f:
        th = line[:-1].split('\t')
        triples1.append([num2id1[int(th[0])], rel2id1[int(th[1])], num2id1[int(th[2])]])


with open(tri2, encoding='utf-8') as f:
    for line in f:
        th = line[:-1].split('\t')
        relations2.append(int(th[1]))
relations2  = list(set(relations2))

for i, rel in enumerate(relations2):
    id2rel2[i]  = rel
    rel2id2[rel]  = i

with open(tri2, encoding='utf-8') as f:
    for line in f:
        th = line[:-1].split('\t')
        triples2.append([num2id2[int(th[0])], rel2id2[int(th[1])], num2id2[int(th[2])]])

with open(link, encoding ='utf-8') as f:
    for line in f:
        th = line[:-1].split('\t')
        links.append([num2id1[int(th[0])],num2id2[int(th[1])]])


with open(file=feature_path, mode='r', encoding='utf-8') as f:
    embedding_list = json.load(f)
    print(len(embedding_list), 'rows,', len(embedding_list[0]), 'columns.')

with open(nid_path1, mode = 'w',encoding='utf-8') as f:
    i = 0
    for e in entity1:
        f.write(str(i)+'\t'+e+'\n')
        i = i + 1
with open(nid_path2, mode = 'w',encoding='utf-8') as f:
    i = 0
    for e in entity2:
        f.write(str(i)+'\t'+e+'\n')
        i = i + 1
with open(ntri1, mode = 'w',encoding='utf-8') as f:
    for tr in triples1:
        f.write(str(tr[0])+'\t' +str(tr[1])+ '\t' + str(tr[2]) + '\n')


with open(ntri2, mode = 'w',encoding='utf-8') as f:
    for tr in triples2:
        f.write(str(tr[0])+'\t' +str(tr[1])+ '\t' + str(tr[2]) + '\n')

with open(nlink, mode = 'w',encoding='utf-8') as f:
    for lin in links:
        f.write(str(lin[0]) + '\t' + str(lin[1])+'\n')

embeddingdict1 = {}
embeddingdict2 = {}
for e in range(len(entity1)):
    embeddingdict1[e] = embedding_list[e]
for e in range(len(entity2)):
    embeddingdict2[e] = embedding_list[e+len(entity1)]

json_str1 = json.dumps(embeddingdict1)
with open(nembedding1, mode = 'w',encoding='utf-8') as f:
    f.write(json_str1)

json_str2 = json.dumps(embeddingdict2)
with open(nembedding2, mode = 'w',encoding='utf-8') as f:
    f.write(json_str2)

# with open(nembedding1, mode = 'r',encoding='utf-8') as f:
#     a = json.load(f)
p = 1

