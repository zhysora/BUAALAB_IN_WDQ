from os.path import join
import os
import numpy as np
import keras.backend as K
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from utils.cache import LMDBClient
from utils import data_utils
from utils import settings

LMDB_NAME = "author_100.emb.weighted" #(pid-j, x^-)
lc = LMDBClient(LMDB_NAME) # 作者 特者 嵌入 加权 平均 (x^-)

data_cache = {}


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def root_mean_log_squared_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), np.inf) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), np.inf) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))


def create_model(): # 创建 神经 网络 
    model = Sequential() # input(300， 100) -> 双向LSTM(64) -> Dense(1)
    model.add(Bidirectional(LSTM(64), input_shape=(300, 100))) 
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(loss="msle",
                  optimizer='rmsprop',
                  metrics=[root_mean_squared_error, "accuracy", "msle", root_mean_log_squared_error])

    return model


def sampler(clusters, k=300, batch_size=10, min=1, max=300, flatten=False): # 从clusters(aid, pid-j)中 取样
    xs, ys = [], [] 
    for b in range(batch_size):
        num_clusters = np.random.randint(min, max) # 随机 设置簇大小 是取样簇大小， 真实的簇大小
        sampled_clusters = np.random.choice(len(clusters), num_clusters, replace=False) # 随机 取 不放回 取的是簇 (aid, pid-j)
        items = []
        for c in sampled_clusters: # 抽取的 簇
            items.extend(clusters[c]) # 列表拼接 将选取的簇的文档都混在一起 
        sampled_points = [items[p] for p in np.random.choice(len(items), k, replace=True)] # 随机取 放回 取的是 文档 pid-j
        x = []
        for p in sampled_points:
            if p in data_cache: # x 中 放入 文档p 的 嵌入 特征x^-
                x.append(data_cache[p])
            else:
                print("a")
                x.append(lc.get(p))
            #print(np.array(x[-1]).shape)
        if flatten:
            xs.append(np.sum(x, axis=0))
        else: # 堆积后 加入 xs
            xs.append(np.stack(x))
        ys.append(num_clusters) # y是对应 簇大小 标记 
    return np.stack(xs), np.stack(ys) # 再做 堆积


def gen_train(clusters, k=300, batch_size=1000, min=1, max=300, flatten=False):
    while True:
        yield sampler(clusters, k, batch_size, min, max, flatten=flatten)


def gen_test(k=300, flatten=False): # 测试集 中 抽样 k个
    name_to_pubs_test = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_test_100.json') # 测试集 name->aid->pid-j
    #xs = []
    xs, ys = [], [] # 特征 与 标记
    names = []
    for name in name_to_pubs_test: # 枚举名字 对于一个名字name 有重复抽样k个 文档
        names.append(name) # 加入 名字列表 中
        num_clusters = len(name_to_pubs_test[name]) # name 下 的 真实 聚类数
        x = [] # 在name下 抽样k个 文档特征x^- 放入一个列表中
        items = []
        ''' for item in name_to_pubs_test[name]: # 属于他的 文档id
            items.append(item) '''
        for c in name_to_pubs_test[name]:  # one person 对于 name下 的 一个 实体 c
            for item in name_to_pubs_test[name][c]: # 属于他的 文档id
                items.append(item) # 加入 到 文档列表 中 
        sampled_points = [items[p] for p in np.random.choice(len(items), k, replace=True)] # 文档集 中 随机取样 k 个
        for p in sampled_points: 
            if p in data_cache: # 在 cache 中 
                x.append(data_cache[p]) # 从 cache 中 取出 特征x^-
            else:
                x.append(lc.get(p)) # 否则 从 数据库 中 取出 特征x^-
        if flatten:
            xs.append(np.sum(x, axis=0))
        else: # 条件走的是 这里
            xs.append(np.stack(x)) # 数组堆积后 放入 xs
            ys.append(num_clusters) # ys 存标记 即 实际聚类大小
    xs = np.stack(xs) # 再堆积一次 此时 xs = array([ [(一个name下的若干文档) [100维特征向量(x^-)], ... ], [[...], ...], ...]) 
    ys = np.stack(ys) # ys = array([聚类大小1, 聚类大小2...])
    return names, xs, ys # 姓名name， 文档特征(x^-)， 聚类大小


def run_rnn(k=300, seed=1106):
    name_to_pubs_train = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_train_500.json') # 训练集 name->aid->pid-j
    #test_names, test_x, test_y = gen_test(k=300)
    # test_names, test_x, test_y = gen_test(k) # 测试集 姓名name， 文档特征集(x^-)， 聚类大小
    np.random.seed(seed) # 随机种子
    clusters = [] # (aid, pid-j)
    for domain in name_to_pubs_train.values(): 
        for cluster in domain.values(): # 作者实体 aid
            clusters.append(cluster)
    for i, c in enumerate(clusters): # 作者实体 aid
        if i % 100 == 0:
            print(i, len(c), len(clusters))
        for pid in c: # 文档id
            data_cache[pid] = lc.get(pid) # 从数据库 中 得到 嵌入x^- 放入到 cache 中
            try:
                if not lc.get(pid).any():
                    print("OH NO") 
            except AttributeError as e:
                print(pid)
    model = create_model()
    # print(model.summary())
    model.fit_generator(gen_train(clusters, k=300, batch_size=100, min=1, max=300), steps_per_epoch=1, epochs=1)
    ''' model.fit_generator(gen_train(clusters, k=300, batch_size=1000), steps_per_epoch=100, epochs=1000,
                        validation_data=(test_x, test_y)) # fit_generator 是一种节省内存 的训练方式 '''
    model_json = model.to_json() # 将模型结构 保存为json
    model_dir = join(settings.OUT_DIR, 'model') #json保存路径
    os.makedirs(model_dir, exist_ok=True) #创建保存路径
    with open(join(model_dir, 'model-count.json'), 'w') as wf: #创建文件并保存
        wf.write(model_json)
    model.save_weights(join(model_dir, 'model-count.h5')) #保存模型 的权重
    
''' 
    kk = model.predict(test_x) # 测试集的 预测 值
    
    name_to_ncluster = {}
    for i, name in enumerate(test_names):
        name_to_ncluster[name] = str(int(kk[i][0]))
    data_utils.dump_json(name_to_ncluster, settings.OUT_DIR, 'n_clusters_rnn.json', True)    

    wf = open(join(settings.OUT_DIR, 'n_clusters_rnn.txt'), 'w') # 写入 文件
    for i, name in enumerate(test_names):
        wf.write('{}\t{}\t{}\n'.format(name, test_y[i], kk[i][0])) # 名字 真实值 预测大小
    wf.close()  '''


if __name__ == '__main__':
    run_rnn()
