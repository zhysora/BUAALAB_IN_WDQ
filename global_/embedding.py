import logging
from os.path import join
from gensim.models import Word2Vec
import numpy as np
import random
from utils.cache import LMDBClient
from utils import data_utils
from utils.data_utils import Singleton
from utils import settings

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) #设置 日志 
EMB_DIM = 100 


@Singleton
class EmbeddingModel:

    def __init__(self, name="aminer"):
        self.model = None
        self.name = name

    def train(self, wf_name, size=EMB_DIM): #训练
        data = []
        LMDB_NAME = 'pub_authors.feature' # 用author_feature.txt 导入到 到 数据库
        lc = LMDBClient(LMDB_NAME) #连接 数据库 (pid-j, author_feature)
        author_cnt = 0
        with lc.db.begin() as txn:
            for k in txn.cursor(): #通过cursor 遍历
                author_feature = data_utils.deserialize_embedding(k[1]) #从k[1]中  反序列化  得到作者特征对象 
                if author_cnt % 10000 == 0:
                    print(author_cnt, author_feature[0])
                author_cnt += 1 #计算作者总数
                random.shuffle(author_feature) #打乱 作者特征
                # print(author_feature)
                data.append(author_feature) #加入 数据集 data 中
        self.model = Word2Vec(
            data, size=size, window=5, min_count=5, workers=20,
        )# 输入字符集， 词向量维数， 窗口大小（当前词与目标词的最大距离）， 词频过滤值， 训练的并行 
        self.model.save(join(settings.EMB_DATA_DIR, '{}.emb'.format(wf_name))) #训练结果的保存 至aminer.emb 

    def load(self, name):
        self.model = Word2Vec.load(join(settings.EMB_DATA_DIR, '{}.emb'.format(name)))
        return self.model

    def project_embedding(self, tokens, idf=None): #输入特征集features， idf字典 {feature: idf}
        """
        weighted average of token embeddings
        :param tokens: input words
        :param idf: IDF dictionary
        :return: obtained weighted-average embedding
        """
        if self.model is None:
            self.load(self.name)
            print('{} embedding model loaded'.format(self.name))
        vectors = [] # 向量集
        sum_weight = 0 # 权值和
        for token in tokens: # 枚举 一个 特征单词 
            if not token in self.model.wv: #如果 单词 不在 Word2Vec 模型 中， 跳过
                continue
            weight = 1
            if idf and token in idf: # token 能在 idf字典 中 查到
                weight = idf[token] # 取出 对应 idf值
            v = self.model.wv[token] * weight # 取出Word2Vec 中 对应 到 向量， 并乘以权重
            vectors.append(v) # 加入到向量集中
            sum_weight += weight # 计算总权重和
        if len(vectors) == 0:
            print('all tokens not in w2v models')
            # return np.zeros(self.model.vector_size)
            return None
        emb = np.sum(vectors, axis=0) #将向量集 中 到 每个向量都加起来
        emb /= sum_weight # 处以总的权重和
        return emb # 返回的 是一个嵌入 d维的向量 在Word2Vec得到100维向量映射后 利用idf加权平均后 x^-


if __name__ == '__main__':
    wf_name = 'aminer'
    emb_model = EmbeddingModel.Instance()
    emb_model.train(wf_name)
    print('loaded')
