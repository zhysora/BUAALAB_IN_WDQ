from os.path import join
import os
import codecs
import math
from collections import defaultdict as dd
from global_.embedding import EmbeddingModel
from datetime import datetime
from utils.cache import LMDBClient
from utils import data_utils
from utils import feature_utils
from utils import string_utils
from utils import settings
from scripts.preprocessing import dump_author_features_to_cache, cal_feature_idf, dump_author_embs
from global_.prepare_local_data import dump_inter_emb, gen_local_data
from local.gae.train import gae_for_na
from keras.models import Model, model_from_json
import numpy as np

def process_by_name(pubs_dict):
    ### preprocessing
    print('n_papers: ', len(pubs_dict)) 
    
    wf = codecs.open(join(settings.GLOBAL_DATA_DIR, 'author_features.txt'), 'w', encoding='utf-8')
    pids = []

    for paper in pubs_dict:
        if "title" not in paper or "authors" not in paper:
            continue
        
        pids.append(paper['id'])
        title_features, keywords_features, venue_features, abst_features, fields_features = feature_utils.extract_common_features(paper)
        name_feature = []
        org_features = []
        for coauthor in paper["authors"]:
            coauthor_name = coauthor.get("name", "")
            coauthor_org = string_utils.clean_name(coauthor.get("org", ""))
            if len(coauthor_name) > 2: 
                name_feature.extend(
                    feature_utils.transform_feature([string_utils.clean_name(coauthor_name)], "name")
                )
            if len(coauthor_org) > 2:
                org_features.extend(
                    feature_utils.transform_feature(string_utils.clean_sentence(coauthor_org.lower()), "org")
                )
        author_feature = name_feature + org_features + title_features + keywords_features + venue_features + abst_features + fields_features

        aid = paper['id']
        wf.write(aid + '\t' + ' '.join(author_feature) + '\n')
    wf.close()

    dump_author_features_to_cache()
    emb_model = EmbeddingModel.Instance()
    emb_model.train('aminer')
    cal_feature_idf()
    dump_author_embs()

    ### prepare_local_data
    IDF_THRESHOLD = 32
    dump_inter_emb(pids)
    gen_local_data(idf_threshold=IDF_THRESHOLD, pids=pids)

    ### count_size
    LMDB_NAME = "author_100.emb.weighted" #(pid-j, x^-)
    lc = LMDBClient(LMDB_NAME) # 作者 特者 嵌入 加权 平均 (x^-)

    k = 300
    test_x = []
    x = [] # 在name下 抽样k个 文档特征x^- 放入一个列表中
    sampled_points = [pids[p] for p in np.random.choice(len(pids), k, replace=True)] # 文档集 中 随机取样 k 个
    for p in sampled_points: 
        x.append(lc.get(p)) # 否则 从 数据库 中 取出 特征x^-
    test_x.append(np.stack(x))
    test_x = np.stack(test_x)

    model_dir = join(settings.OUT_DIR, 'model') #设定模型目录
    rf = open(join(model_dir, 'model-count.json'), 'r') # 加载模型结构
    model_json = rf.read()
    rf.close()
    loaded_model = model_from_json(model_json)
    loaded_model.load_weights(join(model_dir, 'model-count.h5')) # 加载模型 权重
    
    kk = loaded_model.predict(test_x)

    ### local\gae\train
    ret = gae_for_na('Name', int(kk[0][0]))
    return ret
    
def rem_dir(path):
    for i in os.listdir(path):
        path_file = os.path.join(path,i)  
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            for f in os.listdir(path_file):  
                path_file2 = os.path.join(path_file,f)
            if os.path.isfile(path_file2):
                os.remove(path_file2)

if __name__ == '__main__':
    start_time = datetime.now()
    
    tot_pub = 0
    tot_aid = 0

    test_pubs_dict = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'test_pubs_raw.json')
    for name in test_pubs_dict:
        tot_aid = tot_aid + 1
        tot_pub = tot_pub + len(test_pubs_dict[name])
        rem_dir(join(settings.DATA_DIR, 'emb'))
        rem_dir(join(settings.DATA_DIR, 'lmdb'))
        
        ret = process_by_name(test_pubs_dict[name])
        print(ret)
    
    print('finish all')
    print('time:', datetime.now()-start_time)
    print('tot pusbs:', tot_pub)
    print('tot names:', tot_aid)