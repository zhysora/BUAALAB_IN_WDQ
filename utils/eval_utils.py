import numpy as np
from keras import backend as K
from sklearn.metrics import roc_auc_score


def pairwise_precision_recall_f1(preds, truths): # 预测标记， 真实标记
    tp = 0
    fp = 0
    fn = 0
    n_samples = len(preds)
    for i in range(n_samples - 1):
        pred_i = preds[i]
        for j in range(i + 1, n_samples):
            pred_j = preds[j]
            if pred_i == pred_j: # 预测标记 认为 i与j 一致
                if truths[i] == truths[j]: # 真实标记 认为 i与j 一致
                    tp += 1
                else:
                    fp += 1
            elif truths[i] == truths[j]:
                fn += 1
    tp_plus_fp = tp + fp
    tp_plus_fn = tp + fn
    if tp_plus_fp == 0:
        precision = 0.
    else:
        precision = tp / tp_plus_fp
    if tp_plus_fn == 0:
        recall = 0.
    else:
        recall = tp / tp_plus_fn

    if not precision or not recall:
        f1 = 0.
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def cal_f1(prec, rec):
    return 2*prec*rec/(prec+rec)


def get_hidden_output(model, inp): #获得隐藏层输出
    # K.function 实例化 Keras 函数， 输入张量列表， 输出张量列表； 
    # K.learning_phase 返回训练模式/测试模式的flag
    # model.inputs[:1] 这里取出的是第0维 应该是 emb_anchor
    # model.layers[5].get_output_at(0) layers[5]应该是stacked_dist 取出的是pos_dist的输出
    get_activations = K.function(model.inputs[:1] + [K.learning_phase()], [model.layers[5].get_output_at(0), ])
    activations = get_activations([inp, 0]) # 将测试集丢进去 设置为训练模式0
    return activations[0] #?_? 这里返回什么


def predict(anchor_emb, test_embs):
    score1 = np.linalg.norm(anchor_emb-test_embs[0]) # 求第二范数
    score2 = np.linalg.norm(anchor_emb-test_embs[1])
    return [score1, score2]


def full_auc(model, test_triplets): #评估模型函数， 返回AUC 值; 传入参数： 被评估模型， 测试集
    """
    Measure AUC for model and ground truth on all items.

    Returns:
    - float AUC
    """

    grnds = [] #真实值
    preds = [] #预测值
    preds_before = [] #预测值之前的
    embs_anchor, embs_pos, embs_neg = test_triplets #取出anchor嵌入层， positive嵌入层， negative嵌入层 x^-

    inter_embs_anchor = get_hidden_output(model, embs_anchor)  # 取出嵌入层在内部的中间值 理论上他应该要获得的是 y^-
    inter_embs_pos = get_hidden_output(model, embs_pos) 
    inter_embs_neg = get_hidden_output(model, embs_neg) 
    # print(inter_embs_pos.shape)

    accs = [] #分数值
    accs_before = [] #之前的分数值

    for i, e in enumerate(inter_embs_anchor): #枚举 内部的 anchor
        if i % 10000 == 0:
            print('test', i)

        emb_anchor = e # 分别取出对应模型计算后的 第 i 个
        emb_pos = inter_embs_pos[i]
        emb_neg = inter_embs_neg[i]
        test_embs = np.array([emb_pos, emb_neg]) # 转换成数组

        emb_anchor_before = embs_anchor[i] # 取出 初始时 第i个
        emb_pos_before = embs_pos[i]
        emb_neg_before = embs_neg[i]
        test_embs_before = np.array([emb_pos_before, emb_neg_before])

        predictions = predict(emb_anchor, test_embs) # anchor-pos 和 anchor-neg 的距离
        predictions_before = predict(emb_anchor_before, test_embs_before)

        acc_before = 1 if predictions_before[0] < predictions_before[1] else 0 #anchor离 pos更近 则为 1
        acc = 1 if predictions[0] < predictions[1] else 0
        accs_before.append(acc_before) # accuracy 准确率 
        accs.append(acc)

        grnd = [0, 1] # 最理想的是 pos_dist = 0, neg_dist = 1
        grnds += grnd
        preds += predictions
        preds_before += predictions_before

    auc_before = roc_auc_score(grnds, preds_before) # 传入实际标签， 预测值； 返回AUC分数
    auc = roc_auc_score(grnds, preds)
    print('test accuracy before', np.mean(accs_before)) # 取平均
    print('test accuracy after', np.mean(accs))

    print('test AUC before', auc_before)
    print('test AUC after', auc)
    return auc
