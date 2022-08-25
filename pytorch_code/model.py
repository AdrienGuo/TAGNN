import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F

import ipdb
import wandb


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)  # 在建立 item embeddings 的時候用到
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size, bias=False)  #target attention
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        """ 處理 session embedding (local embedding, global embedding)
            & 算分數
        Args:
            hidden: 應該是算出來的 item embeddings ()
        """
        ##########################################################
        # local embedding
        ##########################################################
        # ht: 應該是最後一個 item embedding
        # 也就是 local embedding
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size

        ##########################################################
        # global embedding
        # 公式 6, 7
        ##########################################################
        # q1: 上面的 ht 再經過一個 linear 轉換??
        # self.linear_one 應該是對應到公式 6 的 W_{1}
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size

        # self.linear_two 應該是對應到公式 6 的 W_{2}
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size

        # alpha: 對應到公式 6 的 alpha
        # self.linear_three 對應到公式 6 的 q^{T}
        # alpha = torch.sigmoid(alpha) # B,S,1
        alpha = self.linear_three(torch.sigmoid(q1 + q2))  # (b,s,1)
        # SR-GNN 裏面的 alpha 沒有經過 softmax
        alpha = F.softmax(alpha, 1) # B,S,1

        # a: 應該就是 global embedding 了
        # 對應到公式 6 的 s_{g}
        # 至於他到底怎麼乘出來的，先不管他 XD
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)  # (b,d)
        # 看要不要使用 hybrid
        # 有使用的話對應到公式 7 的 s_{h} (h: hybrid)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))

        ##########################################################
        # 算分數
        # 公式 8, 9
        # 但是公式 9 怎麼沒看到 softmax (似乎只要使用 cross entropy loss 就會有使用到 softmax)
        ##########################################################
        # b: 對應到公式 8 的 v_{i}，但我不懂怎麼對應起來的
        b = self.embedding.weight[1:]  # n_nodes x latent_size

        # 和 SR-GNN 不一樣的地方
        # 有加入 target attention
        # target attention: sigmoid(hidden M b)
        # mask  # batch_size x seq_length
        hidden = hidden * mask.view(mask.shape[0], -1, 1).float()  # batch_size x seq_length x latent_size
        qt = self.linear_t(hidden)  # batch_size x seq_length x latent_size
        # beta = torch.sigmoid(b @ qt.transpose(1,2))  # batch_size x n_nodes x seq_length
        beta = F.softmax(b @ qt.transpose(1,2), -1)  # batch_size x n_nodes x seq_length
        target = beta @ hidden  # batch_size x n_nodes x latent_size
        a = a.view(ht.shape[0], 1, ht.shape[1])  # b,1,d
        a = a + target  # b,n,d

        # scores: 對應到公式 8 的 z^{hat}
        scores = torch.sum(a * b, -1)  # b,n
        # scores = torch.matmul(a, b.transpose(1, 0))

        return scores

    def forward(self, inputs, A):
        """ 處理 item embedding
            使用 GNN 來做處理

        Return:
            hidden: item embeddings
        """
        # 這個 embedding 是在一開始宣告 SessionGraph 就會產生，存放的是所有的 item embedding (n_node, hidden_size)
        # 這裡的 inputs 是有經過 “排序” 的 items, ex: [0, 5, 7, 11, 0, 0]
        # 產生的 hidden 即是上面那些 items 的 item embeddings (有排序)

        # 所以的確我一開始想的沒錯，這些 item embedding 會一直不斷的更新??
        hidden = self.embedding(inputs)

        # 再把 A, hidden 丟到 GNN 裏面，得到運算過後的 item embeddings
        hidden = self.gnn(A, hidden)

        return hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    """
    Args:
        model: SessionGraph
    """
    # 只會根據這個 slice 的情況來取得這些
    # 所以每次的維度會不一樣 (主要是因為在 data.get_slice() 裏面的 max_n_node 的大小不一致)
    # (第一個維度都會一樣是 batchSize)
    alias_inputs, A, items, mask, targets = data.get_slice(i)

    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())

    ##########################################################
    # 計算 item embeddings
    # GNN 就是在 model 裡面計算的
    ##########################################################
    hidden = model(items, A)

    ##########################################################
    # 將 item embeddings 排列成 “原始” 的時序 (需要參考 alias_inputs)
    # ex:
    # alias_inputs: [1, 3, 2, 0, 0, 0]
    # [0, 5, 7, 11, 0, 0] -> [5, 11, 2, 0, 0, 0]
    ##########################################################
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])

    ##########################################################
    # 真正去算出預測的結果
    # session embeddings 也寫在裡面
    ##########################################################
    # model.compute_scores(): 計算出最後預測出來的機率值
    # (但並不是真的代表機率值，而且之後 pytorch 在做 cross entropy 的時候還會再經過 softmax)
    return targets, model.compute_scores(seq_hidden, mask)


def train_test(model, train_data, test_data):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))

    average_loss = total_loss / len(slices)
    # 為什麼要印出 total_loss??
    print(f"Total loss: {total_loss:<6.3f} | Average loss: {average_loss:<6.3f}")
    # print('\tLoss:\t%.3f' % total_loss)

    wandb.log({
        "Total loss": total_loss,
        "Average loss": average_loss
    })

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores = forward(model, i, test_data)
        # sub_scores = scores.topk(20)[1]
        sub_scores = scores.topk(100)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr
