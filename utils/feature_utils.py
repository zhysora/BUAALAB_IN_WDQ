from multiprocessing import Pool
from datetime import datetime
from itertools import chain
from utils.cache import LMDBClient
from utils import string_utils
from utils import data_utils


def transform_feature(data, f_name, k=1):#变换特征 输入数据， 特征名
    if type(data) is str:
        data = data.split() #按空白字符分词
    assert type(data) is list #data 是一个 str列表
    features = [] #列表
    for d in data: #枚举d 
        features.append("__%s__%s" % (f_name.upper(), d)) #（特征名的大写， 单词） 中间的%代表格式化操作 __$f_name.upper()$__$d$
    return features #返回的是字符串列表， 每一项是 __$f_name的大写$__$特征单词$ 的形式


def extract_common_features(item): #提取共同特征
    title_features = transform_feature(string_utils.clean_sentence(item["title"], stemming=True).lower(), "title") #将title项提取特征
    keywords_features = []
    keywords = item.get("keywords") #提取 keywords 项
    if keywords:
        keywords_features = transform_feature([string_utils.clean_name(k) for k in keywords], 'keyword') #提取keyword中的单词， 去除连接符
    fields_features = []
    fields = item.get('fields')
    if fields:
        fields_features = transform_feature([string_utils.clean_name(k) for k in fields], 'fields')
    venue_features = [] #提取机构venue项
    venue_name = item.get('venue', '')
    if venue_name:
        if len(venue_name) > 2:
            venue_features = transform_feature(string_utils.clean_sentence(venue_name.lower()), "venue") #小写化 去除分界符 变换特征
    abst_features = []
    abst = item.get('abst')
    if abst:
        abst_features = transform_feature(string_utils.clean_sentence(abst.lower()), "abst") 
    return title_features, keywords_features, venue_features, abst_features, fields_features


def extract_author_features(item, order=None): #提取作者特征 item中的order
    title_features, keywords_features, venue_features, abst_features, fields_features = extract_common_features(item) #提取共同特征 标题， 关键字， 收录机构
    author_features = []
    for i, author in enumerate(item["authors"]): #枚举第i个作者， author
        if order is not None and i != order: #找到所要的 第order个作者 
            continue
        name_feature = [] #姓名特征
        org_features = [] #机构特征
        org_name = string_utils.clean_name(author.get("org", "")) #格式化机构名 按".", "-", " "分割 小写化
        if len(org_name) > 2:
            org_features.extend(transform_feature(org_name, "org")) #列表加列表
        for j, coauthor in enumerate(item["authors"]): #枚举 合作者
            if i == j: 
                continue
            coauthor_name = coauthor.get("name", "") #获得名字
            coauthor_org = string_utils.clean_name(coauthor.get("org", "")) #获得格式化机构名
            if len(coauthor_name) > 2: 
                name_feature.extend(
                    transform_feature([string_utils.clean_name(coauthor_name)], "name") #格式化 与 特征变换
                ) #将合作者名特征加入 名字特征中
            if len(coauthor_org) > 2:
                org_features.extend(
                    transform_feature(string_utils.clean_sentence(coauthor_org.lower()), "org") #格式化 与 特征变换
                ) #将合作者机构特征加入 机构特征中
        author_features.append(
            name_feature + org_features + title_features + keywords_features + venue_features + abst_features + fields_features
        ) #将以上特征 都 整合 到 作者特征 中 
    author_features = list(chain.from_iterable(author_features)) #创建 为 迭代器 列表 
    return author_features # 到这里 就是把 各个特征 对应的单词列表 合并到一个列表里了
