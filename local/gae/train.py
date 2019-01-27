from __future__ import division
from __future__ import print_function

import os
import time
from os.path import join

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import codecs
import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from local.gae.optimizer import OptimizerAE, OptimizerVAE
from local.gae.input_data import load_local_data
from local.gae.model import GCNModelAE, GCNModelVAE
from local.gae.preprocessing import preprocess_graph, construct_feed_dict, \
    sparse_to_tuple, normalize_vectors, gen_train_edges, cal_pos_weight
from utils.cluster import clustering
from utils.data_utils import load_json, dump_json
from utils.eval_utils import pairwise_precision_recall_f1, cal_f1
from utils import settings

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')  # 32
flags.DEFINE_integer('hidden2', 64, 'Number of units in hidden layer 2.')  # 16
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'gcn_vae', 'Model string.') # 使用的模型 是 gcn_vae
flags.DEFINE_string('name', 'hui_fang', 'Dataset string.')
# flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('is_sparse', 0, 'Whether input features are sparse.')

model_str = FLAGS.model
name_str = FLAGS.name
start_time = time.time()

name_to_ncluster = load_json(settings.OUT_DIR, 'n_clusters_rnn.json')    


def gae_for_na(name, n_clusters): # 对一个具体的姓名预测其消歧结果  评估值[pre, rec, f1], 文档数， 聚类数
    """
    train and evaluate disambiguation results for a specific name
    :param name:  author name
    :return: evaluation results
    """
    adj, features, pids = load_local_data(name=name) # 邻接矩阵(i,j)=1， 文档特征集(y, aid), 标记集 aid索引编号i; features与labels关于下标 一一对应

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros() # 在原邻接矩阵的基础上 去对角线元素 删除0
    adj_train = gen_train_edges(adj) # 这里搞了半天 感觉就是 把adj的对角元素删了 用的csr_matrix 类型 

    adj = adj_train # 完整的邻接矩阵 

    # Some preprocessing
    adj_norm = preprocess_graph(adj)  # 标准化 矩阵 A^' 返回的是元组tuple 坐标(x,y), 值, 形状
    num_nodes = adj.shape[0] # 节点数
    input_feature_dim = features.shape[1] # 输入 特征 维数 [0]是个数
    if FLAGS.is_sparse:  # TODO to test
        # features = sparse_to_tuple(features.tocoo())
        # features_nonzero = features[1].shape[0]
        features = features.todense()  # TODO
    else: # 条件 进入的是 这边
        features = normalize_vectors(features)# 特征向量 标准化

    # Define placeholders
    # tf.placeholder 此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值 ?_?
    placeholders = {
        # 'features': tf.sparse_placeholder(tf.float32),
        'features': tf.placeholder(tf.float32, shape=(None, input_feature_dim)),
        'adj': tf.sparse_placeholder(tf.float32),# 为稀疏张量插入占位符，该稀疏张量将始终被提供
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())# 该函数将返回一个张量。与 input 具有相同的类型。一个占位符张量，默认为 input 的占位符张量 (如果未送入)。
    }

    # Create model
    model = None
    if model_str == 'gcn_ae':
        model = GCNModelAE(placeholders, input_feature_dim)
    elif model_str == 'gcn_vae':# 使用的模型 是 gcn_vae
        model = GCNModelVAE(placeholders, input_feature_dim, num_nodes)
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()  # negative edges/pos edges
    print('positive edge weight', pos_weight)# 负边/正边
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.nnz) * 2) # 矩阵中非零元素的数量nnz

    # Optimizer
    with tf.name_scope('optimizer'):
        if model_str == 'gcn_ae':
            opt = OptimizerAE(preds=model.reconstructions,
                              labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                          validate_indices=False), [-1]),
                              pos_weight=pos_weight,
                              norm=norm)
        elif model_str == 'gcn_vae':# 使用的模型 是 gcn_vae
            opt = OptimizerVAE(preds=model.reconstructions,
                               labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                           validate_indices=False), [-1]),
                               model=model, num_nodes=num_nodes,
                               pos_weight=pos_weight,
                               norm=norm)

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    adj_label = adj_train + sp.eye(adj_train.shape[0])# sp.eye 单位矩阵 标记， 解码应该得到原矩阵
    adj_label = sparse_to_tuple(adj_label)# 稀疏矩阵 -> 元组 (坐标， 值， 维度形状)

    def get_embs():# 获得内部 嵌入z
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)  # z_mean is better
        return emb

    # Train model
    for epoch in range(FLAGS.epochs):# 训练批次 epoch

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy],
                        feed_dict=feed_dict)

        # Compute average loss
        avg_cost = outs[1]
        avg_accuracy = outs[2]

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
              "train_acc=", "{:.5f}".format(avg_accuracy),
              "time=", "{:.5f}".format(time.time() - t))

    emb = get_embs() # 经过 编码器后 的 嵌入层
    ''' n_clusters = int(name_to_ncluster.get(name, 0))
    n_clusters = len(set(labels))# 直接获得 真实的 聚类大小
    if n_clusters == 1:
        return None, None, None, None '''
    #n_clusters = len(set(labels))# 直接获得 真实的 聚类大小
    emb_norm = normalize_vectors(emb)# 标准化 嵌入层
    clusters_pred = clustering(emb_norm, num_clusters=max(n_clusters,1)) # 聚类， 嵌入集 与 聚类大小

    print('clusters_pred: ', clusters_pred)

    ret = {}
    for i, pred_label in enumerate(clusters_pred):
        pred_label = str(pred_label)
        if pred_label not in ret:
            ret[pred_label] = []
        ret[pred_label].append(pids[i])

    rett = []
    for pred_label in ret:
        tmp = []
        for pid in ret[pred_label]:
            tmp.append(pid)
        rett.append(tmp)
    return rett

    ''' ret = {}
    for i, pred_label in enumerate(clusters_pred):
        pred_label = str(pred_label)
        if pred_label not in ret:
            ret[pred_label] = []
        ret[pred_label].append(pids[i].split('-')[0])
        #if i <= 2 :
        #    print(pred_label, pids[i].split('-')[0])

    prec, rec, f1 =  pairwise_precision_recall_f1(clusters_pred, labels)# 计算评估值 prec, rec, f1
    print('pairwise precision', '{:.5f}'.format(prec),
          'recall', '{:.5f}'.format(rec),
          'f1', '{:.5f}'.format(f1))
    return [prec, rec, f1], num_nodes, n_clusters, ret '''


def load_test_names():
    return load_json(settings.DATA_DIR, 'test_name_list.json')


def main():
    names = load_test_names()# 加载测试 作者名 列表
    ans = {}

    wf = codecs.open(join(settings.OUT_DIR, 'local_clustering_results.csv'), 'w', encoding='utf-8')# 结果保存 文件
    wf.write('name,n_pubs,n_clusters,precision,recall,f1\n')#姓名， 论文数， 聚类数， 准确率， 召回， f1分数
    metrics = np.zeros(3)# 3个0
    cnt = 0
    for name in names:#枚举 姓名
        cur_metric, num_nodes, n_clusters, ans[name] = gae_for_na(name)#评估值[pre, rec, f1], 文档数， 聚类数
        if cur_metric == None:
            continue
        wf.write('{0},{1},{2},{3:.5f},{4:.5f},{5:.5f}\n'.format(# 保存到文件
            name, num_nodes, n_clusters, cur_metric[0], cur_metric[1], cur_metric[2]))
        wf.flush()
        for i, m in enumerate(cur_metric): # 各评估值 求和 取平均
            metrics[i] += m
        cnt += 1
        macro_prec = metrics[0] / cnt
        macro_rec = metrics[1] / cnt
        macro_f1 = cal_f1(macro_prec, macro_rec)
        print('average until now', [macro_prec, macro_rec, macro_f1]) # 现在的 各宏-评估值， 计算到 当前name的
        time_acc = time.time()-start_time
        print(cnt, 'names', time_acc, 'avg time', time_acc/cnt)# 运算 的 时间
    macro_prec = metrics[0] / cnt
    macro_rec = metrics[1] / cnt
    macro_f1 = cal_f1(macro_prec, macro_rec)
    wf.write('average,,,{0:.5f},{1:.5f},{2:.5f}\n'.format(
        macro_prec, macro_rec, macro_f1))# 最终 的 各宏-评估值
    wf.close() 

    dump_json(ans, settings.OUT_DIR, 'local_clustering_results.json', True) 


if __name__ == '__main__':
    # gae_for_na('philip_kam_tao_li')
    # gae_for_na('hongbin_liang')
    # gae_for_na('j_yu')
    # gae_for_na('s_yu')
    main()
