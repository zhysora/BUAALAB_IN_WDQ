from keras import backend as K


def l2Norm(x): #l2_normalization 标准化  x_i = x_i/norm(x) norm(x)=sqrt(sum(x)^2)
    return K.l2_normalize(x, axis=-1) #这个函数的作用是利用 L2 范数对指定维度  进行标准化 各维度变换到0~1之间 


def euclidean_distance(vects): #计算欧几里得距离
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def triplet_loss(_, y_pred): # 损失函数
    margin = K.constant(1)
    return K.mean(K.maximum(K.constant(0), K.square(y_pred[:,0,0]) - K.square(y_pred[:,1,0]) + margin)) #max(pos_dist - neg_dist + margin)
 

def accuracy(_, y_pred): # 度量函数
    return K.mean(y_pred[:,0,0] < y_pred[:,1,0]) # pos_dist < neg_dist
