import math

import nn
from sentence_transformers import models, SentenceTransformer

model_name='imxly/sentence_rtb3'
word_embedding_model=models.Transformer(model_name)
pooling_model=models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                             pooling_mode_mean_tokens=True,
                             pooling_mode_cls_token=False,
                             pooling_mode_max_tokens=False)
model=SentenceTransformer(modules=[word_embedding_model,pooling_model])

def evaluate(model,s1,s2):
    #计算余弦相似度
    import numpy as np
    v1=model.encode(s1)
    v2=model.encode(s2)
    v1=v1/np.linalg.norm(v1)
    v2=v2/np.linalg.norm(v2)
    return v1.dot(v2)

s1="小明在吃苹果"
s2="有一个人在吃苹果"

print(evaluate(model,s1,s2))

#计算词向量之间的欧式距离
def get_distance(vec1,vec2):
    sum=0
    for i in range(len(vec1)):
        temp_dist=vec1[i]-vec2[i]
        pow_temp_dist=math.pow(temp_dist,2)
        sum+=pow_temp_dist
    dist=math.sqrt(sum)
    return dist

#triplet loss
import torch.nn.functional as F
class TripletLoss(nn.Module):
    def __init__(self,margin):
        super(TripletLoss,self).__init__()
        self.margin=margin

    def forward(self,anchor,positve,negtive,size_average=True):
        distance_positive=get_distance(anchor,positve)
        distance_negtive=get_distance(anchor,negtive)
        losses=F.relu(distance_positive-distance_negtive+self.margin)
        return losses.mean() if size_average else losses.sum()
#mean()是求取均值
