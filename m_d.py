"""
    Description: m_d

"""
import math
import ast
import os
import networkx as nx
import numpy as np
import torch

from utils.utils import text_to_list, save_pickle, load_pickle


# 计算TF(word代表被计算的单词，word_list是被计算单词所在文档分词后的字典)
def tf(word, word_list):
    return word_list.get(word) / sum(word_list.values())


# 统计含有该单词的句子数
def count_sentence(word, wordcount):
    return sum(1 for i in wordcount if i.get(word))


# 计算IDF
def idf(word, wordcount):
    return math.log(len(wordcount) / (count_sentence(word, wordcount) + 1))


# 计算TF-IDF
def tfidf(word, word_list, wordcount):
    tf = word_list.get(word) / sum(word_list.values())
    idf = math.log(len(wordcount) + 1 / (count_sentence(word, wordcount) + 1))
    tf_idf = tf * idf
    return tf_idf

current_path = os.path.abspath(__file__)
father_path = os.path.abspath(os.path.dirname(current_path))
dxy_dataset = os.path.join(father_path, 'KGData', 'dataset_cmd')
sym_path = os.path.join(dxy_dataset, 'symptoms_cmd.txt')
sym = text_to_list(sym_path)
dis_path = os.path.join(dxy_dataset, 'diseases_cmd.txt')
dis = text_to_list(dis_path)
dxy_goal_path = os.path.join(dxy_dataset, 'goal_cmd.pk')
dxy_goal = load_pickle(dxy_goal_path)
d_sym = []
for g in dxy_goal['all']:
    sym_dict = {}
    for s in g['explicit_inform_slots']:
        sym_dict[s] = g['explicit_inform_slots'][s]
    for s in g['implicit_inform_slots']:
        sym_dict[s] = g['implicit_inform_slots'][s]
    d_sym.append(sym_dict)
with open(os.path.join(dxy_dataset, 'dise_sym_num_high_dict(≥3).txt'), 'r', encoding='utf-8') as f:
# with open(os.path.join(dxy_dataset, 'dise_sym_num_dict_infalse_dxy.txt'), 'r', encoding='utf-8') as f:
    content = f.readlines()
    dis_sym_num = ast.literal_eval(content[0])

tfidf_matrix = torch.zeros((len(dis), len(sym)))
tfidf_matrix_ = torch.zeros((len(dis), len(sym)))
# d_sym = [dis_sym_num[d] for d in dis_sym_num.keys()]
number = 0
for d in dis:
    for s, v in dis_sym_num[d].items():
        tfidf_matrix[dis.index(d)][sym.index(s)] = tfidf(s, dis_sym_num[d], d_sym)
for i in range(len(dis)):
    for j in range(len(sym)):
        tfidf_matrix_[i][j] = tfidf_matrix[i][j]/sum(tfidf_matrix[i])
# rwr_matrix_path = os.path.join(gmd_dataset, 'gmd_tfidf_matrix_update.txt')
# np.savetxt(rwr_matrix_path, tfidf_matrix_, fmt="%f", delimiter=' ')
# dis_p = np.array([[]])
# print(dis_p.shape)
# print(tfidf_matrix_.shape)
# sym_p = np.matmul(dis_p, tfidf_matrix_)
# print(sym_p)
# print(np.argmax(sym_p))
#
# print(np.argsort(-sym_p))