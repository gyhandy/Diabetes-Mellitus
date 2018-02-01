# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

# 1 select the top-k for each classification model
# 2 combine all of the classification by set the top-k id
# 3 plus the rank number and prob of each setted id
# 4 pd.sort_values of 'rank_sum' of 'prob_sum'

# import data

# # offline
# rf = pd.read_csv('./output/rf_best1_98_offline.csv')
# svm = pd.read_csv('./output/svm_offline.csv')
# xgb = pd.read_csv('./output/xgb_offline.csv')
# # select the top 20 index
# cla_list = [rf, svm, xgb]

# online
xgb_1 = pd.read_csv('/home/ubuntu/PycharmProjects/Tianchi/classification/output/10arg5_2_xgb_59.csv')
xgb_2 = pd.read_csv('/home/ubuntu/PycharmProjects/Tianchi/classification/output/10arg5_2_xgb_80_210.csv')
#select the top 20 index
cla_list = [xgb_1, xgb_1]


pre_id = pd.DataFrame()
for cla in cla_list:
    # normalize the 'prob'
    cla['prob'] = (cla['prob'] - cla['prob'].min()) / (cla['prob'].max() - cla['prob'].min())
    cla_top_20_id = cla['id'][0:20]
    pre_id = pd.concat([pre_id, cla_top_20_id], axis=1, join_axes=[cla_top_20_id.index])
preid = np.array(pre_id)

# # offline
# out_id = np.transpose(preid).reshape(-1, 1)
# only_id = list(set(out_id.flatten().tolist()))
# rank_sum = []
# prob_sum = []
# real_value = []
# for i in range(len(only_id)):
#     each_id_ranks = []
#     each_id_probs = []
#     # only offline
#     # find real value
#     each_id_value = float(cla_list[0]['blood suger'][cla_list[0].id == only_id[i]])
#     for cla in cla_list:
#         rank = cla[cla.id == only_id[i]].index
#         prob = float(cla['prob'][cla.id == only_id[i]])
#         each_id_ranks.append(rank)
#         each_id_probs.append(prob)
#     each_id_ranks_sum = sum(each_id_ranks)
#     each_id_probs_sum = sum(each_id_probs)
#     rank_sum.append(each_id_ranks_sum)
#     prob_sum.append(each_id_probs_sum)
#     real_value.append(each_id_value)
#
# rank_data = pd.DataFrame(only_id, columns=["id"])
# rank_data["rank_sum"] = rank_sum
# rank_data["prob_sum"] = prob_sum
# # only offline
# rank_data["blood suger"] = real_value
# rank_data = rank_data.sort_values(by='rank_sum', axis=0, ascending=True)# get the top-k by 'rank_sum'
# # rank_data = rank_data.sort_values(by='prob_sum', axis=0, ascending=False)# get the top-k by 'rank_sum'
# print(rank_data)
# rank_data.to_csv("./final_top_cla/offline/rf+xgb+svm_offline_rank.csv", index=None, encoding='utf-8')



# online
out_id = np.transpose(preid).reshape(-1, 1)
only_id = list(set(out_id.flatten().tolist()))
rank_sum = []
prob_sum = []
real_value = []
for i in range(len(only_id)):
    each_id_ranks = []
    each_id_probs = []
    for cla in cla_list:
        rank = cla[cla.id == only_id[i]].index
        prob = np.array(cla['prob'][cla.id == only_id[i]])
        each_id_ranks.append(rank)
        each_id_probs.append(prob)
    each_id_ranks_sum = sum(each_id_ranks)
    each_id_probs_sum = sum(each_id_probs)
    rank_sum.append(each_id_ranks_sum)
    prob_sum.append(each_id_probs_sum)

rank_data = pd.DataFrame(only_id, columns=["id"])
rank_data["rank_sum"] = rank_sum
rank_data["prob_sum"] = prob_sum
rank_data = rank_data.sort_values(by='rank_sum', axis=0, ascending=True)
# rank_data = rank_data.sort_values(by='prob_sum', axis=0, ascending=False)
print(rank_data)
rank_data.to_csv("./final_top_cla/online/rf+xgb+svm_online_rank.csv", index=None, encoding='utf-8')


