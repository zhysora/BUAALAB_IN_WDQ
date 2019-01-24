from os.path import join
import codecs
import math
from collections import defaultdict as dd
from global_.embedding import EmbeddingModel
from datetime import datetime
from utils.cache import LMDBClient
from utils import data_utils
from utils import feature_utils
from utils import settings

start_time = datetime.now()


def dump_author_features_to_file(): #提取作者特征到文件中
    """
    generate author features by raw publication data and dump to files
    author features are defined by his/her paper attributes excluding the author's name
    """
    pubs_dict = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'pubs_raw.json') #原始数据 pubs_raw.json
    print('n_papers', len(pubs_dict)) #论文数量
    wf = codecs.open(join(settings.GLOBAL_DATA_DIR, 'author_features.txt'), 'w', encoding='utf-8') #特征写入 author_features.txt
    for i, pid in enumerate(pubs_dict): #枚举一篇论文 i, pid = 索引， 枚举对象
        if i % 1000 == 0:
            print(i, datetime.now()-start_time)
        paper = pubs_dict[pid] # 某个paper 的信息
        if "title" not in paper or "authors" not in paper:
            continue
        if len(paper["authors"]) > 30: # 合作者 人数
            print(i, pid, len(paper["authors"]))
        if len(paper["authors"]) > 100:
            continue
        n_authors = len(paper.get('authors', [])) #该论文的作者数 dict.get(key, default=None) 在字典中查询键值key 若不存在返回默认值default
        for j in range(n_authors): #枚举每一位作者
            if 'id' not in paper['authors'][j]:
                continue
            author_feature = feature_utils.extract_author_features(paper, j) #提取论文paper中的作者j的特征 __$f_name$_$word$
            aid = '{}-{}'.format(pid, j) #aid: pid-j
            wf.write(aid + '\t' + ' '.join(author_feature) + '\n') #往wf中写入特征信息 aid\t author_feature\n
    wf.close()


def dump_author_features_to_cache(): #将作者特征 导入到 cache 中 本地数据库 lmdb
    """
    dump author features to lmdb 
    """
    LMDB_NAME = 'pub_authors.feature'
    lc = LMDBClient(LMDB_NAME)
    with codecs.open(join(settings.GLOBAL_DATA_DIR, 'author_features.txt'), 'r', encoding='utf-8') as rf: #之前把特征写入到了文件 auther_features.txt 中， 这里 读取
        for i, line in enumerate(rf): #枚举 第i行 line 1行对应一个author_feature  pid-j\tauthor_feature
            if i % 1000 == 0:
                print('line', i)
            items = line.rstrip().split('\t') #删除末尾空格 后  按'\t'分割  pid-j, author_feature
            pid_order = items[0] #提取文档序号 对应 上一个函数中的输出格式  pid-j 文档id-第j个作者
            author_features = items[1].split() # 提取作者特征 每个特征用空格分割为 列表了
            lc.set(pid_order, author_features) #导入 到 数据库 中


def cal_feature_idf(): #计算逆文档频率
    """
    calculate word IDF (Inverse document frequency) using publication data
    """
    feature_dir = join(settings.DATA_DIR, 'global')  #特征目录
    counter = dd(int) # 一种字典， 比{}多一个 如果没有查询到的key， 会返回int(0)
    cnt = 0
    LMDB_NAME = 'pub_authors.feature' # (pid-j, author_feature)
    lc = LMDBClient(LMDB_NAME) #连接 lmdb
    author_cnt = 0
    with lc.db.begin() as txn:
        for k in txn.cursor(): #遍历 lmdb
            features = data_utils.deserialize_embedding(k[1]) #反序列化 得到 特征对象 k[0]是id, k[1]是author_feature
            if author_cnt % 10000 == 0:
                print(author_cnt, features[0], counter.get(features[0])) #features[0] 是 类似"__NAME__yanjun_zhang" 是合作者的name_feature
            author_cnt += 1 #作者计数
            for f in features:
                cnt += 1 #记总数
                counter[f] += 1 # 记特征f 的出现次数
    idf = {}
    for k in counter: # 计算特征k 对应的 idf
        idf[k] = math.log(cnt / counter[k]) 
    data_utils.dump_data(dict(idf), feature_dir, "feature_idf.pkl") #写入 feature_idf.pkl 中 {feature: idf}


def dump_author_embs():# 将作者嵌入 导入到 lmdb 中，  作者嵌入 是  词向量 IDF 的 加权平均
    """
    dump author embedding to lmdb
    author embedding is calculated by weighted-average of word vectors with IDF
    """
    emb_model = EmbeddingModel.Instance()
    idf = data_utils.load_data(settings.GLOBAL_DATA_DIR, 'feature_idf.pkl') #取出 上个函数 计算的 idf {feature: idf}
    print('idf loaded') 
    LMDB_NAME_FEATURE = 'pub_authors.feature' # (pid-j, author_feature)
    lc_feature = LMDBClient(LMDB_NAME_FEATURE) # 连接 作者特征 lmdb
    LMDB_NAME_EMB = "author_100.emb.weighted" # (pid-j, x^-)
    lc_emb = LMDBClient(LMDB_NAME_EMB) # 连接 作者嵌入 lmdb
    cnt = 0
    with lc_feature.db.begin() as txn:
        for k in txn.cursor(): # 遍历 特征 
            if cnt % 1000 == 0: 
                print('cnt', cnt, datetime.now()-start_time)
            cnt += 1
            pid_order = k[0].decode('utf-8') # 解码获得 文章 编号
            features = data_utils.deserialize_embedding(k[1]) # 反序列化 得 对应 作者特征 对象
            cur_emb = emb_model.project_embedding(features, idf) # 获得 对应 加权平均IDF 的 嵌入 x^-
            if cur_emb is not None:
                lc_emb.set(pid_order, cur_emb) # 结果 保存 到 作者 嵌入lmdb author_100.emb.weigthed 中  (pid-j, x^-)
            else:
                print(pid_order)


if __name__ == '__main__':
    """
    some pre-processing
    """
    dump_author_features_to_file() # 将作者特征写入本地文件 中 
    dump_author_features_to_cache() # 将作者特征写入 cache 中 
    emb_model = EmbeddingModel.Instance() # 实例化 嵌入 模型 
    emb_model.train('aminer')  # training word embedding model Word2Vec
    cal_feature_idf() # 计算 特征 到 逆文档频率
    dump_author_embs() # 将作者嵌入 导入 数据库 中
    print('done', datetime.now()-start_time)
