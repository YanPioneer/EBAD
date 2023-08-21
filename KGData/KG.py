import ast

import torch
import os


class KGADJ:

    def __init__(self, device):

        self.disease_pre = []
        self.sys_pre = []
        self.dis_sys_pre = {}
        self.device = device

        # self.kg_matrix = torch.eye(130, 130).to(self.device)  # 构建邻接矩阵
        self.kg_matrix = torch.zeros(385, 385).to(self.device)  # 构建邻接矩阵
        # print(self.kg_matrix)
        # print(self.kg_matrix.size())

        # self.initialize_adj()

        current_path = os.path.abspath(__file__)
        father_path = os.path.abspath(os.path.dirname(current_path))
        data_path = os.path.join(father_path, 'dataset_cmd')

        with open(os.path.join(data_path, 'diseases_cmd.txt'), 'r', encoding='utf-8') as d:
            content = d.readlines()
            for line in content:
                info = line.strip().split('\t')
                self.disease_pre.append(info[0])
        # print(self.disease_pre)

        with open(os.path.join(data_path, 'symptoms_cmd.txt'), 'r', encoding='utf-8') as d:
            content = d.readlines()
            for line in content:
                info = line.strip().split('\t')
                self.sys_pre.append(info[0])
        # print(self.sys_pre)

        # with open(os.path.join(data_path, 'dise_sym_num_dict.txt'), 'r', encoding='utf-8') as f:
        with open(os.path.join(data_path, 'dise_sym_num_high_dict(≥3).txt'), 'r', encoding='utf-8') as f:
            content = f.readlines()
            self.dis_sys_pre = ast.literal_eval(content[0])
            # print(self.dis_sys_pre)

        with open(os.path.join(data_path, 'dise_sym_num_dict.txt'), 'r', encoding='utf-8') as f:
            content = f.readlines()
            self.dis_sys_all_pre = ast.literal_eval(content[0])
            # print(self.dis_sys_all_pre)

        self.disease_ = self.disease_pre
        self.sys_ = self.sys_pre
        self.dis_sys_ = self.dis_sys_pre

    def initialize_adj(self):
        self.kg_matrix = torch.zeros(385, 385).to(self.device)  # 构建邻接矩阵
        # 构建上三角、下三角
        for i in range(len(self.disease_pre)):
            for j in range(len(self.sys_pre)):
                if self.sys_pre[j] in list(self.dis_sys_pre[self.disease_pre[i]].keys()):
                    # print(self.sys_pre[j], self.disease_pre[i])
                    self.kg_matrix[i][j + len(self.disease_pre)] = 1
                    self.kg_matrix[j + len(self.disease_pre)][i] = 1

        # print(self.kg_matrix.size()[0])
        # for i in range(self.kg_matrix.size()[0]):
        #     print(self.kg_matrix[i])

    def update_adj(self, confirm_symptoms):  # 根据患者存在的症状进行疾病排除
        del_dis = []
        if len(confirm_symptoms) > 0:  # 确定要删除的疾病
            for sym in confirm_symptoms:
                for d in list(self.dis_sys_all_pre.keys()):
                    if sym not in list(self.dis_sys_all_pre[d].keys()):
                        del_dis.append(d)

        for d_ in del_dis:
            # 更改索引
            self.kg_matrix[self.disease_pre.index(d_), :] = 0
            self.kg_matrix[:, self.disease_pre.index(d_)] = 0
            # self.kg_matrix[self.disease_pre.index(d_), self.disease_pre.index(d_)] = 1


