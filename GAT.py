import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)

class GAT(nn.Module):

    """
    Thanks to https://github.com/Diego999/pyGAT
    """
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.W = nn.Linear(768,nfeat)
        self.out = nn.Linear(nfeat,1)
        self.attentions = [GraphAttentionLayer(nfeat, nhid,dropout=dropout, alpha=alpha, concat=True,tag = True) for _ in
                           range(nheads)] #768 --> 128
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False,tag = False)

    def forward(self, x,tag,offset_mapping,attention_mask, adj):
        x_offset = torch.zeros_like(x).to(x.device)
        index_group = torch.zeros_like(offset_mapping).to(x.device)
        for i in range(offset_mapping.shape[0]):
            index=0
            index_start = 0
            index_end = 1
            for k in range(offset_mapping.shape[1]):
                if offset_mapping[i][k].tolist() == [0,0]:
                    break
                if offset_mapping[i][k][1] == offset_mapping[i][k+1][0]-1:
                    x_offset[i][index] = torch.mean(x[i][index_start:index_end],dim=0) #mean vaule will be taken for subwaords
                    index_group[i][index] = torch.tensor([index_start, index_end])
                    index += 1
                    index_start = index_end
                    index_end +=1
                else:
                    index_end +=1
        input_x = x_offset #B,S,D
        x = self.W(input_x)
        #GAT process
        x = torch.cat([att(x,tag,attention_mask, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)

        x = F.elu(self.out_att(x, tag,attention_mask,adj))
        x = self.out(x)
        x = x.squeeze(dim=2)

        attention_mask = torch.where(tag>0,attention_mask,torch.zeros_like(attention_mask))
        x = torch.mul(x,attention_mask)
        x = F.softmax(torch.where(attention_mask > 0, x, -9e10 * torch.ones_like(x)),dim=1)
        #pre_define length and step
        input_index = torch.arange(start=128,end = 0, step=-1).to(x.device)
        summary = torch.sum(attention_mask,dim=1).to(x.device)
        input_index = torch.div(input_index,summary.unsqueeze(1).repeat([1,128]))
        input_index = torch.mul(input_index, attention_mask).to(torch.float32)
        input_index = F.softmax(
            torch.where(attention_mask > 0, input_index, -9e10 * torch.torch.ones_like(input_index)), dim=1)
        #score = representation score + position score
        x = x + input_index
        #sort score
        x = torch.sort(input=x, dim=1, descending=True,stable=True).indices
        ec = index_group.clone()
        #position exchange
        index_group_fix = ec.scatter_(dim=1,index = x.unsqueeze(2).repeat([1,1,2]),src=index_group)
        offset_mapping_fix = torch.zeros_like(offset_mapping)
        index_fix = torch.arange(128,dtype=torch.long).unsqueeze(0).repeat([offset_mapping_fix.shape[0],1])
        for i in range(offset_mapping_fix.shape[0]):
            l=0
            for k in range(128):
                if index_group_fix[i][k].tolist() == [0,0]:
                    break
                else:
                    for j in range(int(index_group_fix[i][k][0]),int(index_group_fix[i][k][1])):
                        index_fix[i][l] = j
                        l+=1
        return index_fix


class GraphAttentionLayer(nn.Module):
    """
    Thanks to https://github.com/Diego999/pyGAT
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, tag, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.tag = tag
        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Linear(2 * out_features, 1)


        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input,tag,attention_mask, adj):

        h = self.W(input)  # 8,128,128
        N = h.size()[1]

        a_input = torch.cat([h.repeat(1, 1, N).view(h.shape[0], N * N, -1), h.repeat(1, N, 1)], dim=2) \
            .view(h.shape[0], N, -1, 2 * self.out_features)  # 8,128,128,256
        e = self.leakyrelu(self.a(a_input).squeeze(3))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # (BatchSize,Seq,Seq)
        attention = F.softmax(attention, dim=2)  # (BatchSize,Seq,Seq)
        h_prime = []
        for per_a, per_h in zip(attention, h):
            h_prime.append(torch.matmul(per_a, per_h))

        h_prime = torch.stack(h_prime)

        if self.concat:

            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
