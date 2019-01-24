import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

flags = tf.app.flags
FLAGS = flags.FLAGS


def sparse_to_tuple(sparse_mx): # 稀疏矩阵 -> 元组 (坐标， 值， 维度形状)
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose() # 提取元素坐标 [[x1, y1], [x2, y2] ...]
    values = sparse_mx.data # 矩阵元素中的值 讲道理都是1
    shape = sparse_mx.shape # 维度 讲道理是(节点数， 节点数)
    return coords, values, shape # 坐标(x,y), 值, 形状


def normalize_vectors(vectors): # 标准化 向量集
    scaler = StandardScaler()
    vectors_norm = scaler.fit_transform(vectors) # 去均值和方差归一化。且是针对每一个特征维度来做的，而不是针对样本。 
    return vectors_norm 


def preprocess_graph(adj):  # use original version, adj not contain diags 
    adj = sp.coo_matrix(adj) # 变成 坐标存储稀疏矩阵 adj 没对角元素 但是补全为对称矩阵了 即(i, j) <=> (j, i)
    adj_ = adj + sp.eye(adj.shape[0]) # sp.eye()是单位阵 adj_就是adj基础上 加上对角线 全1
    rowsum = np.array(adj_.sum(1)) # 按行 求和 就是求点i的度数
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten()) # 度数矩阵D 对角矩阵 i的度数的 ^(-0.5) 幂  
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo() # A^' = D^T A D
    return sparse_to_tuple(adj_normalized) # 坐标(x,y), 值, 形状


def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict


def gen_train_edges(adj): # 输入 邻接矩阵 是个稀疏矩阵 坐标形式存储
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape) 
    adj.eliminate_zeros() # 去对角线 元素 并 去0
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj) # 获得 上三角 元素 
    adj_tuple = sparse_to_tuple(adj_triu) # 稀疏矩阵 -> 元组 (坐标， 值， 维度形状)
    edges = adj_tuple[0] # 获得坐标 (i, j)
    data = np.ones(edges.shape[0]) # 每条边都是1 
    adj_train = sp.csr_matrix((data, (edges[:, 0], edges[:, 1])), shape=adj.shape) 
    adj_train = adj_train + adj_train.T
    return adj_train # 完整的邻接矩阵 无对角元素


def cal_pos_weight(adj):
    pos_edges_num = adj.nnz
    return (adj.shape[0] * adj.shape[0] - pos_edges_num) / pos_edges_num
