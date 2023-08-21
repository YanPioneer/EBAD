import torch
from torch import nn as nn


class LabelAttn(nn.Module):
    def __init__(self, attention_dim, label_num, embed_size, sym_num, device):
        super(LabelAttn, self).__init__()
        self.model_name = 'LabelAttn'
        self.device = device
        self.sym_representation = nn.Embedding(sym_num, embed_size)
        self.liner_project = nn.Linear(embed_size, attention_dim)
        self.attn_score = nn.Linear(attention_dim, label_num)

    def forward(self, mask_matrix):  # mask_matrix: n X
        """
        :param x: n x embed_size
        :param mask_matrix: n x embed_size 0 for no 1 for yes
        :return:
        """
        x = torch.tensor([i for i in range(358)]).to(self.device)
        x_embed = self.sym_representation(x)
        x_embed = mask_matrix * x_embed
        Z = torch.tanh(self.liner_project(x_embed))           # batch_size x sym_num x attention_dim
        A = torch.softmax(self.attn_score(Z), dim=1)      # batch_size x sym_num x |L|
        V = torch.bmm(A.transpose(1, 2), x_embed)             # batch_size x |L| x (2 * hidden_dim)
        # disease_dis = torch.softmax(torch.sum(V, dim=2))
        disease_ = torch.sum(V, dim=2)

        return disease_


class DiseaseInference(nn.Module):
    def __init__(self, attention_dim, label_num, embed_size, sym_num, device):
        super(DiseaseInference, self).__init__()

        self.labelAttn = LabelAttn(attention_dim, label_num, embed_size, sym_num, device)

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, label, mask):
        disease_ = self.labelAttn(mask)
        loss = self.loss_func(disease_, label)
        # print(torch.softmax(disease_, dim=1))
        output_ = disease_.max(1)[1]
        output = 0
        for i in range(len(output_)):
            if output_[i] == label[i]:
                output += 1
        return output, loss

    def predict(self, mask):
        disease_ = self.labelAttn(mask)
        disease_dis = torch.softmax(disease_, dim=1)
        # print(disease_dis)
        return disease_dis

