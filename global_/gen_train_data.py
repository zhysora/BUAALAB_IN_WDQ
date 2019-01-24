# -*- coding: utf-8 -*-

from os.path import join
import os
import multiprocessing as mp
import random
from datetime import datetime
from utils.cache import LMDBClient
from utils import data_utils
from utils import settings

LMDB_NAME = "author_100.emb.weighted" #连接数据库， 取出作者的特征嵌入， （Word2Vec的加权平均）(pid-j, x^-)
lc = LMDBClient(LMDB_NAME)
start_time = datetime.now()

#为训练全局模型， 生成三元组的训练集
"""
This class generates triplets of author embeddings to train global model
"""


class TripletsGenerator:
    name2pubs_train = {} #训练集原始数据
    name2pubs_test = {} #测试集原始数据
    names_train = None #训练集 中的 所有 作者姓名 集
    names_test = None #测试集 中的 所有 作者姓名 集
    n_pubs_train = None #训练集 论文数量
    n_pubs_test = None #测试集 论文数量
    pids_train = [] #训练集下 的论文集
    pids_test = [] #测试集下 的论文集
    n_triplets = 0 #三元组数量
    batch_size = 100000

    def __init__(self, train_scale=10000): #构造函数
        self.prepare_data() #数据预处理 主要是把json文件中的东西 整合到 相关类成员 中 主要是pids_train 与 pids_test
        self.save_size = train_scale
        self.idf = data_utils.load_data(settings.GLOBAL_DATA_DIR, 'feature_idf.pkl') #取出idf 数据集 {feature: idf}

    def prepare_data(self): #数据预处理
        self.name2pubs_train = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_train_500.json')  # name->aid->pid-j
        self.name2pubs_test = None
        self.names_train = self.name2pubs_train.keys() #返回所有的键值 即 所有的作者名
        print('names train', len(self.names_train)) # 训练集 作者名 数
        self.names_test = None
        #print('names test', len(self.names_test)) # 测试集 作者名 数
        #assert not set(self.names_train).intersection(set(self.names_test)) #训练集 与 测试集 的作者名字 不能有交集
        for name in self.names_train: # 枚举训练集中的 姓名name
            name_pubs_dict = self.name2pubs_train[name] # 取出 训练集中 name下的 作者实体字典 {aid: pid-j}
            for aid in name_pubs_dict: # 枚举同姓名的作者聚类 的一个实体 aid
                self.pids_train += name_pubs_dict[aid] #将aid下的论文集pid-j 放到 pids_train 列表中
        random.shuffle(self.pids_train) #随机打乱
        self.n_pubs_train = len(self.pids_train) # 文档数
        print('pubs2train', self.n_pubs_train) # 训练集 论文数量

        ''' for name in self.names_test: # 这里对 测试集 做 类似的 操作
            #self.pids_test += self.name2pubs_test[name]
            name_pubs_dict = self.name2pubs_test[name]
            for aid in name_pubs_dict:
                self.pids_test += name_pubs_dict[aid] 
        random.shuffle(self.pids_test)
        self.n_pubs_test = len(self.pids_test)
        print('pubs2test', self.n_pubs_test) '''

    def gen_neg_pid(self, not_in_pids, role='train'): #生产negative文档  not_in_pids 不能在这个论文集中 在构建三元组的过程中， 枚举了一个实体aid的论文集pids
        if role == 'train': #取出相应论文集
            sample_from_pids = self.pids_train
        else:
            sample_from_pids = self.pids_test
        while True: #随机 找一片 不再目标论文集 中 的 论文， 并返回
            idx = random.randint(0, len(sample_from_pids)-1)
            pid = sample_from_pids[idx]
            if pid not in not_in_pids:
                return pid

    def sample_triplet_ids(self, task_q, role='train', N_PROC=8): #从论文集中抽样 生成三元组， 元素是文档id表达， 放入task_q中
        n_sample_triplets = 0 # 抽样的 三元组
        if role == 'train': # 根据对应角色， 取出相应的 作者姓名集 与 原始数据集
            names = self.names_train
            name2pubs = self.name2pubs_train  # name->aid->pid-j
        else:  # test
            names = self.names_test
            name2pubs = self.name2pubs_test
            self.save_size = 200000  # test save size
        for name in names: # 枚举 作者姓名 name
            name_pubs_dict = name2pubs[name] # 取出 name 下的 字典 {aid: pid-j}
            for aid in name_pubs_dict: # 枚举 一个 作者 实体 aid
                pub_items = name_pubs_dict[aid] # aid 下的 所有 论文集 pid-j
                if len(pub_items) == 1: # 只有1篇 论文 则跳过
                    continue
                pids = pub_items # aid 下的 所有 论文集
                cur_n_pubs = len(pids) # 论文集大小
                random.shuffle(pids) # 随机打乱
                for i in range(cur_n_pubs): # 枚举aid 的一篇论文 i
                    pid1 = pids[i]  # pid 论文i的paper id (pid-j)

                    # batch samples 批抽样
                    n_samples_anchor = min(6, cur_n_pubs) #抽样数量
                    idx_pos = random.sample(range(cur_n_pubs), n_samples_anchor) #在1～cur_n_pubs中  抽取 n_samples_anchor个数
                    for i, i_pos in enumerate(idx_pos): #枚举 样本索引集 idx_pos
                        if i_pos != i: #  论文i作为anchor 考虑另一篇抽取的不同的论文 i_pos 将其作为positive
                            if n_sample_triplets % 100 == 0:
                                # print('sampled triplet ids', n_sample_triplets)
                                pass #占位空语句
                            pid_pos = pids[i_pos] #取出论文 i_pos 对应 的 论文pid-j
                            pid_neg = self.gen_neg_pid(pids, role) # 生成一个论文作为 negative
                            n_sample_triplets += 1 # 构成三元组
                            task_q.put((pid1, pid_pos, pid_neg)) #放入多进程队列中 anchor, positive, nagetive

                            if n_sample_triplets >= self.save_size: #数据达到预设规模 放入N_PROC个 (None, None, None) 这里是为了 给消费者进程 提供终止条件
                                for j in range(N_PROC):
                                    task_q.put((None, None, None))
                                return
        for j in range(N_PROC):
            task_q.put((None, None, None))

    def gen_emb_mp(self, task_q, emb_q): #根据 文档id三元组队列 task_q， 转化为对应的嵌入表达(x)， 放入emb_q中
        while True: #由于是 多进程， 一直循环在做
            pid1, pid_pos, pid_neg = task_q.get() #从队列中取出一个 三元组 （不放回）
            if pid1 is None: # 三元组已经 全部处理 完毕
                break
            emb1 = lc.get(pid1) # 从数据库中 取出相应 文档的 嵌入x^- (这是之前 提取特征工作后 用Word2Vec作用为100维的向量 再取加权平均后 得到每篇文档的嵌入表达)
            emb_pos = lc.get(pid_pos)
            emb_neg = lc.get(pid_neg)
            if emb1 is not None and emb_pos is not None and emb_neg is not None: #检验 合理性， 合理就放入 对应三元组的 嵌入表达 队列emb_q中
                emb_q.put((emb1, emb_pos, emb_neg))
        emb_q.put((False, False, False)) # 搞完后放一个 (False, False, False) 在队列中

    def gen_triplets_mp(self, role='train'): #像一个迭代器， 每次返回一个 嵌入表达x^- 的 三元组
        N_PROC = 8 # 进程数量设置 
        # 多进程队列 （用于进程通信，资源共享）
        task_q = mp.Queue(N_PROC * 6) 
        emb_q = mp.Queue(1000)
        # 创建子进程 target=要执行的方法 args=对应方法需要传入的参数
        producer_p = mp.Process(target=self.sample_triplet_ids, args=(task_q, role, N_PROC)) #生产者进程 构建三元组放入task_q中 triplets 这里三元组的元素是 文档id
        consumer_ps = [mp.Process(target=self.gen_emb_mp, args=(task_q, emb_q)) for _ in range(N_PROC)] #消费者进程 这里是将生产者构建出的三元组中的 文档id 替换成对应的 嵌入表达(x^-) 放入到emb_q中
        producer_p.start() # 开启生产者进程
        [p.start() for p in consumer_ps] #开启所有消费者进程

        cnt = 0

        while True:
            if cnt % 1000 == 0:
                print('get', cnt, datetime.now()-start_time)
            emb1, emb_pos, emb_neg = emb_q.get() # 从队列emb_q取出 嵌入表达（x^-）的三元组 (不放回取出)
            if emb1 is False: #读到一个 False 这是消费者进程 最后结束的 时候放的
                producer_p.terminate() #关闭生产者进程
                producer_p.join() #阻塞当前进程， 直到等待生产者进程 执行完毕
                [p.terminate() for p in consumer_ps] #对消费者进程们 做类似的 事
                [p.join() for p in consumer_ps]
                break
            cnt += 1 #三元组 计数
            yield (emb1, emb_pos, emb_neg) #yield 的作用就是把一个函数变成一个 generator, 每次执行到 yield时，函数就返回一个迭代值，下次迭代时再接着做

    def dump_triplets(self, role='train'): 
        triplets = self.gen_triplets_mp(role) #得到 嵌入表达(x^-) 的三元组集 使用了多进程 multi_process
        if role == 'train': #设定输出目录
            out_dir = join(settings.OUT_DIR, 'triplets-{}'.format(self.save_size)) 
        else:
            out_dir = join(settings.OUT_DIR, 'test-triplets')
        os.makedirs(out_dir, exist_ok=True) #创建目录
        anchor_embs = []
        pos_embs = []
        neg_embs = []
        f_idx = 0
        for i, t in enumerate(triplets): #枚举 三元组t
            if i % 100 == 0:
                print(i, datetime.now()-start_time)
            emb_anc, emb_pos, emb_neg = t[0], t[1], t[2] #取出对应的嵌入向量x^-
            anchor_embs.append(emb_anc) #依次加入到对应列表中
            pos_embs.append(emb_pos)
            neg_embs.append(emb_neg)
            if len(anchor_embs) == self.batch_size: #到达了设定的批次规模， 批次写入到文件中
                data_utils.dump_data(anchor_embs, out_dir, 'anchor_embs_{}_{}.pkl'.format(role, f_idx)) #若干个x^-
                data_utils.dump_data(pos_embs, out_dir, 'pos_embs_{}_{}.pkl'.format(role, f_idx))
                data_utils.dump_data(neg_embs, out_dir, 'neg_embs_{}_{}.pkl'.format(role, f_idx))
                f_idx += 1 #批次计数
                anchor_embs = [] #及时清空
                pos_embs = []
                neg_embs = []
        if anchor_embs: #如果还有剩的， 把最后剩的一批也输出出去
            data_utils.dump_data(anchor_embs, out_dir, 'anchor_embs_{}_{}.pkl'.format(role, f_idx))
            data_utils.dump_data(pos_embs, out_dir, 'pos_embs_{}_{}.pkl'.format(role, f_idx))
            data_utils.dump_data(neg_embs, out_dir, 'neg_embs_{}_{}.pkl'.format(role, f_idx))
        print('dumped') 


if __name__ == '__main__':
    data_gen = TripletsGenerator(train_scale=1000000) #读取相应数据 与 预处理工作 主要是pids_train 与 pids_test
    data_gen.dump_triplets(role='train') #将生成 的 （初始嵌入表达x）三元组 按批次的 输出到文件中
    # data_gen.dump_triplets(role='test')
