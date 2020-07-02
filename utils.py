import networkx as nx
import numpy as np


def build_graph(train_data):
    # 创建有向图
    graph = nx.DiGraph()
    # 遍历训练集中的每个session
    for seq in train_data:
        # 遍历session中的每个item
        for i in range(len(seq) - 1):
            # 如果当前图中没有(seq[i, seq[i+1]])的edge，则weight=1
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            # else, edge存在，则其本身的weight+1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            # 添加edge
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    # 遍历graph中的每个node, 得到normalized weight
    # 使用的edge终点的入度而非起点的出度normalize???
    for node in graph.nodes:
        sum = 0
        # 射入node的edge为(j, i), j为edge的起始node，i为结束node
        for j, i in graph.in_edges(node):
            # 累加各个edge的weight
            sum += graph.get_edge_data(j, i)['weight']
        # weight的normalization
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph


def data_masks(all_usr_pois, item_tail):
    # 收集每一个session的长度
    us_lens = [len(upois) for upois in all_usr_pois]
    # 取所有session最大的长度
    len_max = max(us_lens)
    # 将所有ipois补全到最大长度,用item_tail，即0来补全
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    # mask：对于session中不是补全的item, mask为1，补全的item，mask为0
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    # shuffle
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, shuffle=False, graph=None):
        # data[0]为X
        inputs = data[0]
        # 输入为inputs，补全位为[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        # data[1]为Y
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

    def generate_batch(self, batch_size):
        if self.shuffle:
            # shuffled_arg是一个arr，代表每个session的索引
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        # Number of batch
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        # 等分成n_batch份, slices:n_batch*batch_size
        # 关于np.split()  https://www.jianshu.com/p/d020afd053bc
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        # 调整最后一份的length
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        # slices中储存的为多个索引list（将索引切分为各个batch），如[[0, 1, 2], [3, 4, 5], ...]
        # slices是一个二维List，存放了一个batch里面的多个session
        return slices

    '''
    u_input函数的用法
    
    该函数是去除数组中的重复数字，并进行排序之后输出。
    >>> np.unique([1, 1, 2, 2, 3, 3])
    array([1, 2, 3])
    >>> a = np.array([[1, 1], [2, 3]])
    >>> np.unique(a)
    array([1, 2, 3])
    Return the unique rows of a 2D array

    >>> a = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]])
    >>> np.unique(a, axis=0)
    array([[1, 0, 0], [2, 3, 4]])
    '''

    def get_slice(self, i):
        # inputs = inputs[i] 是一个batch
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            # n_node: 每一个session里面count(distinct(item))，因为一个session里面可能会有重复的item，相当于去重
            # n_node也是一个list，每一个session有一个n_node
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        for u_input in inputs:
            # A node list of the session graph
            # print("=========u_input==========")
            # print(u_input)
            '''
                这里的u_input还没有去重，仍然是原来的补全了0的session list
            '''
            node = np.unique(u_input)
            # 每一个session里面distinct item的数目不同，这里下面的items即每一个session里面的distinct item
            # 为了让A的维度统一，因为是一个Batch一起算，A是batch里所有session的A矩阵，u_A是单条session的A矩阵
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            # A:n*2n,n为session中distinct item的数目，这里的n即为max_n_node
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                # 如果u_input[i + 1] == 0，说明已经到后面补全的地方了，不用看了，结束这一个session的计算
                if u_input[i + 1] == 0:
                    break
                # np.where用法：https://www.cnblogs.com/massquantity/p/8908859.html
                # np.where(condition) 输出满足条件元素的坐标，以tuple的形式给出
                # 取session中的相连node，并给adjacency matrix赋值

                # print("=======where======")
                # print(node)
                # print(u_input)
                # print(i)
                # print(u_input[i])
                # print(u_input[i + 1])
                # print(np.where(node == u_input[i]))
                # print(np.where(node == u_input[i])[0][0])
                # print(np.where(node == u_input[i + 1]))
                # print(np.where(node == u_input[i + 1])[0][0])
                # == == == =where == == ==
                # [0 158 159]
                # [158 159 159 159   0   0   0   0   0   0   0   0   0   0   0   0]
                # 0
                # 158
                # 159
                # (array([1], dtype=int64),)
                # 1
                # (array([2], dtype=int64),)
                # 2

                '''
                    根据session构建A的思路:
                    假如session为1,2,3,2,4
                    这代表有这些有向边：1->2, 2->3, 3->2, 2->4
                    因此在循环中a[1][2]=1，只是代表出度，而node是用来判断坐标位置的，distinct了
                '''

                u = np.where(node == u_input[i])[0][0]  # 取u_input[i]对应的node标号为u
                v = np.where(node == u_input[i + 1])[0][0]  # 取u_input[i+1]对应的node标号为v
                u_A[u][v] = 1
            # Indegree adjacency matrix
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            # Outdegree adjacency matrix
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            # Concatenate
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            # print("===node===")
            # print(u_input)
            # print(node)
            # print(u_A)
            A.append(u_A)
            # 按照session的顺序，将各个item(node)对应的标号作为list存入alias_inputs
            # u_input是每一条session
            # alias_inputs：不存放item_id，而是存放item在图里对应的坐标
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])

        '''
            A:存放一个batch里所有session的A矩阵
            alias_inputs:不存放item_id，而是存放item在图里对应的坐标
            items:不全了0的坐标序列
            mask:
        '''
        return alias_inputs, A, items, mask, targets
