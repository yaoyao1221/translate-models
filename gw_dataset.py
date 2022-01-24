import pandas as pd
import numpy as np


file_name='E:/nlp/zhushi_data3.csv'
dataset=pd.read_csv(file_name)
data_origin=dataset.iloc[:,1:4]
words=dataset.iloc[:,1:2]
print(data_origin.keys())
# print(len(words))
# words_list=words["words"]
#
# from collections import Counter
# w=Counter(words_list)
# w_order=sorted(w.items(),key=lambda x:x[1],reverse=True)
# print(w_order)
from operator import  itemgetter
from itertools import groupby
dict_grouped=data_origin.groupby("words")
list_grouped=list(dict_grouped)
# print(list_grouped[0])
temp_list=list(list_grouped[0])
print(temp_list)
print(type(temp_list[1]))
print(temp_list[1].keys())
print(temp_list[1].values.tolist())

new_filename='E:/nlp/triplet_prepare.txt'
for i in list_grouped:
    i_dataframe=list(i)
    #是dataframe，将dataframe转换成list
    i_list=i_dataframe[1].values.tolist()
    temp_val=[]
    for j in i_list:
        if j not in temp_val:
            temp_val.append(j)
            with open(new_filename,'a') as file_object:
                file_object.write(str(j)+'\n')
    with open(new_filename, 'a') as file_object:
        file_object.write('\n')






