import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size

        # 将Tensor类型转换为可以学习的Parameter类型
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        # print(self.w_ih)
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        # print(self.b_ih)
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        # Y = WX + b
        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        """
        A -- 该批数据图矩阵的列表, (batch_size, n_node, 2 * n_node)
        hidden -- (batch_size, max_session_len_of_the_batch, latent_size)
        """
        # Equation (1) in the paper
        # print("==========A.shape==========")
        # print(A[:, :, :A.shape[1]].shape) # batch_size * n_node * n_node
        # A[:, :, :A.shape[1]]
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        # A[:, :, A.shape[1]: 2 * A.shape[1]]
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)

        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        # torch.chunk(tensor, chunks, dim=0): 将tensor拆分成指定数量的块
        # gi.chunk(3, 2) 将gi分三块，沿第2维
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        # Equation (2)
        inputgate = torch.sigmoid(i_i + h_i)
        # Equation (3)
        resetgate = torch.sigmoid(i_r + h_r)
        # Equation (4)
        newgate = torch.tanh(i_n + resetgate * h_n)
        # Equation (5)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


# 在SessionGraph里面先过一次GNN，然后在后面接全连接层，进行预测之类的
class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        # ('--nonhybrid', action='store_true', help='only use the global preference to predict')
        self.nonhybrid = opt.nonhybrid

        # torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None,
        #                    norm_type=2, scale_grad_by_freq=False, sparse=False)
        # hidden_size为latent_size, 即embedding_dim
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        # print("===emb===")
        # print(self.embedding.weight[:])
        self.gnn = GNN(self.hidden_size, step=opt.step)

        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        """
        hidden -- (batch_size, seq_length, latent_size)
        mask -- (batch_size, seq_length)
        seq_length -- max session length of all
        """
        # 取最后一个动作
        # print("============mask==============")
        # print(mask.shape)
        # print(mask.shape[0])
        # 一个tuple：存放了两个1*100的tensor，第一个tensor是0到100，第二个tensor是每个session非补全item的个数
        # print((torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1))
        # print(hidden.shape) # torch.Size([100, 16, 100])

        # print("=======hidden=========")
        # print(hidden[0].shape) # torch.Size([16, 100])
        # print(hidden[0])
        # print(hidden[[0,1]].shape)  # torch.Size([2, 16, 100])
        # print(hidden[[0,1]])
        # print(hidden[[0, 1][1]].shape)  # torch.Size([16, 100])
        # print(hidden[[0, 1][1]])
        # print(hidden[torch.tensor([0])].shape) # torch.Size([1, 16, 100])
        # ''' 以后筛选tensor可以这样做 就是一个切片'''
        # print(hidden[torch.tensor([0]), torch.tensor([0])].shape)  # torch.Size([1, 100])

        # ht: 挑出batch里每一个session最后的那个item的embedding，就是一个切片
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        # print(ht.shape) # torch.Size([100, 100])

        # 局部偏好
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        # print("=======q.shape======")
        # print(q1.shape)
        # 全局偏好
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        # print(q2.shape)
        # Attention
        # Equation (6)

        ''' 虽然q1和q2维度不同，但是有广播机制 '''
        # print((q1 + q2).shape) # (batch_size, seq_length, latent_size)
        # print(q1 + q2)
        # print("sigmoid")
        # print(torch.sigmoid(q1 + q2))
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        # print(alpha.shape) # (batch_size, seq_length, 1)

        # 乘mask别的补全的为0，就不用算了
        # print(mask.view(mask.shape[0], -1, 1).shape) # (batch_size, seq_length, 1)
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)

        # print("=====a.shape=====")
        # print(a.shape) # (batch_size, latent_size)

        if not self.nonhybrid:
            # Equation (7)
            a = self.linear_transform(torch.cat([a, ht], 1))
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        # print("===a===")
        # print(a)
        # print("===b===")
        # print(b)
        # Equation (8)
        scores = torch.matmul(a, b.transpose(1, 0))

        # print("========scores.shape=====")
        # print(scores.shape)

        return scores

    def forward(self, inputs, A):
        """
        Forward Embedding & GNN
        """
        # print(inputs)
        hidden = self.embedding(inputs)
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
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    # print("==========")
    # print(alias_inputs)
    # print("begin A")
    # print(np.array(A).shape)
    # A只是一个矩阵
    # (100, 6, 12) A的shape，6*12 A是n*2n的
    # print(A)
    # print("end A")
    # print("=========items===========")
    # print(np.array(items).shape)
    # (100, 6) 这个session里面有6个item
    # print(items)
    # print("=========mask===========")
    # print(np.array(mask).shape)
    # (100, 16) 16是max_sequence，mask用来补齐，方便后面的网络层用
    # print(mask)
    # for item in mask:
    #     print(item)
    # print("=========targets===========")
    # print(np.array(targets).shape)
    # targets就是100个batch里面对应的那个target物品
    # print(targets)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden = model(items, A)
    # print(alias_inputs.size())
    # print(items.size())
    # print("==========hidden_size======")
    # print(hidden.size())
    # 选择这个batch中第i个session对应的item序列
    get = lambda i: hidden[i][alias_inputs[i]]
    # print("====forfor====")
    # for i in torch.arange(len(alias_inputs)).long():
    #     print(get(i))
    #     print(alias_inputs[i])
    #     print(hidden[i][alias_inputs[i]])
    # 将batch中所有的item序列按dim=0叠加
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])

    # print("===size")
    # print(alias_inputs.size())
    # print(hidden.size())
    # print(seq_hidden.size())
    # print(seq_hidden.size())
    ''' compute_scores 是最后一层全连接算出来的softmax值 '''
    return targets, model.compute_scores(seq_hidden, mask)


def train_test(model, train_data, test_data):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    # slices储存train_data分batch的索引
    slices = train_data.generate_batch(model.batch_size)
    # print("==========slices=======")
    # print(slices)
    for i, j in zip(slices, np.arange(len(slices))):
        # print(i)
        # print(j)
        model.optimizer.zero_grad()
        # Forward
        ''' score 是最后一层全连接算出来的softmax值 '''
        targets, scores = forward(model, i, train_data)
        # print(scores)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        # Loss Function

        # label_list = trans_to_cpu(targets).detach().numpy().tolist()
        # print(label_list)
        # for user in scores:
        #     # print(len(trans_to_cpu(user).detach().numpy().tolist()))
        #     # print(trans_to_cpu(user).detach().numpy().tolist())
        #     user = torch.nn.functional.softmax(user)
        #     score_list = trans_to_cpu(user).detach().numpy().tolist()
        #     # print(score_list)
        #     # print(" ".join(str(index) + ":" + str(math.log(math.fabs(score))) for index, score in enumerate(score_list)))

        loss = model.loss_function(scores, targets - 1)
        # Backward
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    # 指定计算模型
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d]' % (j, len(slices)))
        targets, scores = forward(model, i, test_data)
        # torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)
        # Returns the k largest elements of the given input tensor along a given dimension.
        # print("=====scores_before=====")
        # print(scores)
        sub_scores = scores.topk(20)[1]
        # print("=====scores_after=====")
        # print(sub_scores)
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        # 每一个session选出前20个物品作为推荐列表，来计算准确率
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            # print("=====score=====")
            # print(score)
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    # print("=====hit=====")
    # print(hit)
    # print(np.sum(hit))
    # print(len(hit))
    hit = np.mean(hit) * 100
    # print(hit)
    mrr = np.mean(mrr) * 100
    return hit, mrr