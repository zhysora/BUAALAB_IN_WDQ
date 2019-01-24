from os.path import join
import os
import numpy as np
from numpy.random import shuffle
from global_.global_model import GlobalTripletModel
from utils.eval_utils import get_hidden_output
from utils.cache import LMDBClient
from utils import data_utils
from utils import settings

#IDF_THRESHOLD = 32  # small data
IDF_THRESHOLD = 32


def dump_inter_emb(pids): # 从训练的全局模型中 取出 隐藏层， 给局部模型使用
    """
    dump hidden embedding via trained global model for local model to use
    """
    LMDB_NAME = "author_100.emb.weighted" # 连接数据库 这是 作者特征经过Word2Vec处理为100维向量后加权平均后的 嵌入(x^-) (pid-j, x^-)
    lc_input = LMDBClient(LMDB_NAME)
    INTER_LMDB_NAME = 'author_triplets.emb' # (pid-j, y)
    lc_inter = LMDBClient(INTER_LMDB_NAME) # 内层嵌入 数据库 将测试集的作者的新嵌入 y 写入其中
    global_model = GlobalTripletModel(data_scale=1000000) # 实例化一个全局模型
    trained_global_model = global_model.load_triplets_model() # 加载一个训练好的全局模型

    embs_input = []
    for pid in pids:
        cur_emb = lc_input.get(pid)
        if cur_emb is None:
            continue
        embs_input.append(cur_emb)
    embs_input = np.stack(embs_input)
    inter_embs = get_hidden_output(trained_global_model, embs_input)
    for i, pid in enumerate(pids):
        lc_inter.set(pid, inter_embs[i])

    ''' name_to_pubs_test = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_test_100.json') #加载 测试集 name->aid->pid-j
    for name in name_to_pubs_test: # 枚举作者名 
        print('name', name)
        name_data = name_to_pubs_test[name] #{aid: pid-j}
        embs_input = [] # 论文对应嵌入x^- 集
        pids = [] # 论文id集
        #for pid in name_data: # 枚举文档id
        #    cur_emb = lc_input.get(pid) # 取出 文档对应嵌入 x^-
        #    if cur_emb is None:
        #        continue
           # embs_input.append(cur_emb) #添加到 对应 列表中
           # pids.append(pid)
        for i, aid in enumerate(name_data.keys()): # 枚举作者id
            if len(name_data[aid]) < 5:  # n_pubs of current author is too small
                continue
            for pid in name_data[aid]: # 枚举文档id
                cur_emb = lc_input.get(pid) # 取出 文档对应嵌入 x^-
                if cur_emb is None:
                    continue
                embs_input.append(cur_emb) #添加到 对应 列表中
                pids.append(pid)        
        embs_input = np.stack(embs_input) # 堆积 增加一维 ?_?： 这是干嘛
        inter_embs = get_hidden_output(trained_global_model, embs_input) # 获得内层嵌入值 ?_?: 这里可以看作是 y吗？
        for i, pid_ in enumerate(pids): # 枚举 文档
            lc_inter.set(pid_, inter_embs[i]) # 写入到数据库 author_triplets.emb中  (pid-j, y) '''


def gen_local_data(pids, idf_threshold=10): # 对每一个作者名， 生成局部数据， 包括文档特征 与 文档网络； 输入参数是阀值， 也就是相似度高于多少才连边
    """
    generate local data (including paper features and paper network) for each associated name
    :param idf_threshold: threshold for determining whether there exists an edge between two papers (for this demo we set 29)
    """
    idf = data_utils.load_data(settings.GLOBAL_DATA_DIR, 'feature_idf.pkl') # 加载 特征的 idf值 {word: idf}
    INTER_LMDB_NAME = 'author_triplets.emb' # 加载 作者在triplet训练后的 内部嵌入 (pid-j, y)
    lc_inter = LMDBClient(INTER_LMDB_NAME)
    LMDB_AUTHOR_FEATURE = "pub_authors.feature" # 加载 作者 原始 特征 (pid-j, author_feature)
    lc_feature = LMDBClient(LMDB_AUTHOR_FEATURE) 
    graph_dir = join(settings.DATA_DIR, 'local', 'graph-{}'.format(idf_threshold)) # 建立目录， 做好保存局部模型 的工作
    os.makedirs(graph_dir, exist_ok=True)

    name = "Name"

    wf_content = open(join(graph_dir, '{}_pubs_content.txt'.format(name)), 'w')
    shuffle(pids) # 打乱
    
    for pid in pids:
        cur_pub_emb = lc_inter.get(pid) # 获得文档嵌入 y
        if cur_pub_emb is not None:
            cur_pub_emb = list(map(str, cur_pub_emb)) #把cur_pub_emb 转换成字符串 表达
            wf_content.write('{}\t'.format(pid)) # 文档id
            wf_content.write('\t'.join(cur_pub_emb)) # 嵌入 y
            wf_content.write('\t{}\n'.format(pid)) # 作者id
    wf_content.close() # pid-j, y, aid

    # generate network
    n_pubs = len(pids) 
    wf_network = open(join(graph_dir, '{}_pubs_network.txt'.format(name)), 'w') # 作者名 - 网络保存路径 (pid-j, pid-j)
    edges_num = 0
    for i in range(n_pubs-1): # 枚举 文档 i
        author_feature1 = set(lc_feature.get(pids[i])) # 取出 文档i 原始 特征 (pid-j, author_feature)
        for j in range(i+1, n_pubs): # 枚举 后面 点 文档 j
            author_feature2 = set(lc_feature.get(pids[j])) # 取出 文档j 原始 特征
            common_features = author_feature1.intersection(author_feature2) # 提取 公共特征
            idf_sum = 0
            for f in common_features: # 枚举 公共特征 中的 特征f
                idf_sum += idf.get(f, idf_threshold)  # 计算 idf 和
                # print(f, idf.get(f, idf_threshold)) 
            if idf_sum >= idf_threshold: # 和 大于阀值
                wf_network.write('{}\t{}\n'.format(pids[i], pids[j])) # 连边， 写入 图网络 文件中 (pid-j, pid-j)
                edges_num = edges_num + 1
    print('n_egdes', edges_num)
    wf_network.close()

    ''' name_to_pubs_test = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_test_100.json') # 加载 测试集 name->aid->pid-j
    idf = data_utils.load_data(settings.GLOBAL_DATA_DIR, 'feature_idf.pkl') # 加载 特征的 idf值 {word: idf}
    INTER_LMDB_NAME = 'author_triplets.emb' # 加载 作者在triplet训练后的 内部嵌入 (pid-j, y)
    lc_inter = LMDBClient(INTER_LMDB_NAME)
    LMDB_AUTHOR_FEATURE = "pub_authors.feature" # 加载 作者 原始 特征 (pid-j, author_feature)
    lc_feature = LMDBClient(LMDB_AUTHOR_FEATURE) 
    graph_dir = join(settings.DATA_DIR, 'local', 'graph-{}'.format(idf_threshold)) # 建立目录， 做好保存局部模型 的工作
    os.makedirs(graph_dir, exist_ok=True)
    for i, name in enumerate(name_to_pubs_test): # 枚举测试集 中 的名字 name
        print(i, name) 
        cur_person_dict = name_to_pubs_test[name] # {aid: pid-j}
        pids_set = set() # 文档id集合 会去重
        pids = [] # 文档id列表
        pids2label = {} #文档id: 作者id 映射

        # generate content 为每一个作者名 保存 相应到内容 到一个文件中
        wf_content = open(join(graph_dir, '{}_pubs_content.txt'.format(name)), 'w') # 作者名 - 内容保存路径 (pid-j, y, aid)
        for aid in cur_person_dict:
            cur_aid_dict = cur_person_dict[aid] 
            for i, pid in enumerate(cur_aid_dict): 
                pids2label[pid] = aid # 文档 标记
                pids.append(pid) # 加入 到文档集       
        #for pid in cur_person_dict: # 枚举 文档id 
        #    pids2label[pid] = 'NULL' # 文档 标记
        #    #pids.append(pid) # 加入 到文档集
        #    for i, aid in enumerate(cur_person_dict): # 枚举 作者id
        #        items = cur_person_dict[aid] 
        #        if len(items) < 5:
        #            continue
        #        for pid in items: # 枚举 文档id 
        #            pids2label[pid] = aid # 文档 标记
        #            pids.append(pid) # 加入 到文档集       
        shuffle(pids) # 打乱
        
        for pid in pids:
            cur_pub_emb = lc_inter.get(pid) # 获得文档嵌入 y
            if cur_pub_emb is not None:
                cur_pub_emb = list(map(str, cur_pub_emb)) #把cur_pub_emb 转换成字符串 表达
                pids_set.add(pid) # 加入到 文档集
                wf_content.write('{}\t'.format(pid)) # 文档id
                wf_content.write('\t'.join(cur_pub_emb)) # 嵌入 y
                wf_content.write('\t{}\n'.format(pids2label[pid])) # 作者id
        wf_content.close() # pid-j, y, aid

        # generate network
        pids_filter = list(pids_set) # 去重文档列表
        n_pubs = len(pids_filter) # 文档总数
        print('n_pubs', n_pubs)
        wf_network = open(join(graph_dir, '{}_pubs_network.txt'.format(name)), 'w') # 作者名 - 网络保存路径 (pid-j, pid-j)
        edges_num = 0
        for i in range(n_pubs-1): # 枚举 文档 i
            #if i % 10 == 0: 
            #    print(i)
            author_feature1 = set(lc_feature.get(pids_filter[i])) # 取出 文档i 原始 特征 (pid-j, author_feature)
            for j in range(i+1, n_pubs): # 枚举 后面 点 文档 j
                author_feature2 = set(lc_feature.get(pids_filter[j])) # 取出 文档j 原始 特征
                common_features = author_feature1.intersection(author_feature2) # 提取 公共特征
                idf_sum = 0
                for f in common_features: # 枚举 公共特征 中的 特征f
                    idf_sum += idf.get(f, idf_threshold)  # 计算 idf 和
                    # print(f, idf.get(f, idf_threshold)) 
                if idf_sum >= idf_threshold: # 和 大于阀值
                    wf_network.write('{}\t{}\n'.format(pids_filter[i], pids_filter[j])) # 连边， 写入 图网络 文件中 (pid-j, pid-j)
                    edges_num = edges_num + 1
        print('n_egdes', edges_num)
        wf_network.close() '''


''' if __name__ == '__main__':
    dump_inter_emb() # 取出内层嵌入
    gen_local_data(idf_threshold=IDF_THRESHOLD) #生成局部数据 同一作者下的相关信息， 局部链接图
    print('done') '''
