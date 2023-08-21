import torch
from torch import nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)  # shape [N, out_features]
        # h = torch.bmm(input, self.W)  # shape [N, out_features]
        bs = adj.size()[0]
        N = h.size()[1]

        a_input = torch.cat([h.repeat(1, 1, N).view(bs, N * N, -1), h.repeat(1, N, 1)], dim=-1).view(bs, N, -1, 2 * self.out_features) # shape[N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # [N,N,1] -> [N,N]

        ### batch_size
        # e = e.repeat(bs, 1).view(bs, N, N)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)  # [N,N], [N, out_features] --> [N, out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, device, kg_node=46, embed_size=100, dis_num=5):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.embed_size = embed_size
        self.kg_node = kg_node
        self.dis_num = dis_num
        self.device = device
        self.sym_representation = nn.Embedding(self.kg_node, self.embed_size)

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

        self.classifier = nn.Sequential(  #
            nn.Linear(self.embed_size, self.dis_num),
        )
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, label, adj, mask):
        x = torch.tensor([[i for i in range(self.kg_node)] for l in range(adj.size()[0])]).to(self.device)
        x_embed = self.sym_representation(x)
        x = F.dropout(x_embed, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        # x = torch.sum(x, dim=1)
        sym_rep = torch.matmul(mask, x).squeeze(1)
        disease_ = self.classifier(sym_rep)
        # disease_ = torch.sum(disease_, dim=1)
        loss = self.loss_func(disease_, label)
        output_ = disease_.max(1)[1]
        output = 0
        for i in range(len(output_)):
            if output_[i] == label[i]:
                output += 1
        return output, loss

    def predict(self, adj):
        x = torch.tensor([[i for i in range(self.kg_node)] for l in range(adj.size()[0])]).to(self.device)
        x_embed = self.sym_representation(x)
        x = F.dropout(x_embed, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        # return F.log_softmax(x, dim=1)
        return x  # 图谱现存节点表示