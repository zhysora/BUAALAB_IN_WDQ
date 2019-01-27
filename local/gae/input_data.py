from os.path import join
import numpy as np
import scipy.sparse as sp
from utils import settings
from global_.prepare_local_data import IDF_THRESHOLD

local_na_dir = join(settings.DATA_DIR, 'local', 'graph-{}'.format(IDF_THRESHOLD))


def encode_labels(labels): # 作者实体 id
    classes = set(labels) # 作者实体 id 去重
    classes_dict = {c: i for i, c in enumerate(classes)} # 作者实体id 索引编号
    return list(map(lambda x: classes_dict[x], labels)) # labels 中的 实体id 变成 相应的索引编号 相当于 离散化了


def load_local_data(path=local_na_dir, name='cheng_cheng'): #加载本地数据 路径，名字
    # Load local paper network dataset
    print('Loading {} dataset...'.format(name), 'path=', path) 

    idx_features_labels = np.genfromtxt(join(path, "{}_pubs_content.txt".format(name)), dtype=np.dtype(str)) # name的内容内容 (pid-j, y, aid)
    features = np.array(idx_features_labels[:, 1:-1], dtype=np.float32)   # 嵌入y 转为float32
    #labels = encode_labels(idx_features_labels[:, -1]) # aid 作者实体id->索引编号 相当于离散化 将原aid替换成 对应的 索引i
    pids = idx_features_labels[:, 0]

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.str) # pid-j 转化为str
    idx_map = {j: i for i, j in enumerate(idx)} # pid-j:索引编号i 相当于离散化
    edges_unordered = np.genfromtxt(join(path, "{}_pubs_network.txt".format(name)), dtype=np.dtype(str)) # name的网络文件 (pid-j, pid-j)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), # 扁平化展开后， 在map中找到 对应离散化 id 形成lst
                     dtype=np.int32).reshape(edges_unordered.shape) # 再reshape成 2维数组 和原来edges_unordered的shape 一致  (i, j)

    if edges.shape == (2, ):
        edges = edges[np.newaxis, :]

    if edges.shape[0] == 0:
        xs = []
        ys = []
        edges_num = 0
        for i, x in enumerate(idx):
            for j, y in enumerate(idx):
                    xs.append(i)
                    ys.append(j)
                    edges_num = edges_num + 1
        adj = adj = sp.coo_matrix((np.ones(edges_num), (np.array(xs), np.array(ys))), #邻接矩阵 用稀疏矩阵存储， 坐标形式 (i,j)=1
                        shape=(features.shape[0], features.shape[0]), dtype=np.float32)
    else:
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), #邻接矩阵 用稀疏矩阵存储， 坐标形式 (i,j)=1
                        shape=(features.shape[0], features.shape[0]), dtype=np.float32) # shape=(文档总数， 文档总数)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) # 补全邻接矩阵 有(i,j)=1 <=> (j,i)=1

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

    #print(labels)
    return adj, features, pids # 邻接稀疏矩阵坐标存储形式(i,j)=1， 文档特征集y, 标记集 aid索引编号i; features与labels关于下标 一一对应


if __name__ == '__main__':
    #load_local_data(name='zhigang_zeng')
    load_local_data(name="lei yu")