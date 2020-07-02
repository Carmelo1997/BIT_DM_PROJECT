import pickle
import math

def get_dict():
    # ff = h5py.File('user_sess_dict.hdf5', 'r')
    # len_user_sess_dict = ff['all_user_count']['all_user_count'][0]
    # len_count_all_item_set = ff['all_item_count']['all_item_count'][0]
    #
    # return len_user_sess_dict, len_count_all_item_set

    fp = open("./paper_user_sess_dict.txt", "rb+")
    user_sess_dict = pickle.load(fp)

    return user_sess_dict

def build_train_set_txt(user_sess_dict):
    # 训练集：(input, label), 不用user的索引, 单个user：([[sess]], label)
    # 多个user：([[[sess]]], [label])
    input = []
    # label = []
    user_id = []

    fp = open("./paper_all_dict.txt", "rb+")
    (user_dict, item_dict, cate_dict, item_cate_dict) = pickle.load(fp)

    for user in user_sess_dict:
        sess_list_list = user_sess_dict[user]
        input.append(sess_list_list)
        user_id.append(int(user))
        # print('===user%d===' % (user))
        # for sess in sess_list_list:
        #     print(sess)

    return input, user_id, len(user_dict), len(item_dict), len(cate_dict), item_cate_dict

def split_validation(input, user_id, item_cate_dict, valid_portion):
    # 获取整个训练集
    # np.arrange(),构造一个索引序列，序列中存放训练集元素的索引
    # np.np.random.shuffle(index_arr) 打乱这个索引序列，仍是一个序列
    # valid_set_x = [train_set_x[s] for s in sidx[n_train:]]， s只是一个索引，指向训练集
    # n_train代表拆分训练集的比例

    train_set_input = []
    train_set_cate = []
    train_set_label = []
    train_set_user = []

    test_set_input = []
    test_set_cate = []
    test_set_label = []
    test_set_user = []

    all_input = []
    all_user_id = []

    # 把多个sess拆分成单个sess，并带上user
    for index, sess_list_list in enumerate(input):
        user_item = user_id[index]
        n_train = int(math.ceil(len(sess_list_list) * (1. - valid_portion)))
        n_valid = len(sess_list_list) - n_train
        # print("train_val " + str(n_train) + " " + str(n_valid))
        for sess_list in sess_list_list[:n_train]:
            train_set_input.append(sess_list[:-1])
            # 每一个item都对应一个cate是下面这个写法
            # train_set_cate.append([item_cate_dict[item] for item in sess_list[:-1]])
            # 下面是把这个sess里面的cate放到set里面去，只统计出现的cate，顺序就不重要了，再转换成list
            temp_set = set([])
            for item in sess_list[:-1]:
                temp_set.add(item_cate_dict[item])
            train_set_cate.append(list(temp_set))
            train_set_label.append(sess_list[-1])
            train_set_user.append(user_item)
        if n_valid > 0:
            for sess_list in sess_list_list[-n_valid:]:
                test_set_input.append(sess_list[:-1])
                # 每一个item都对应一个cate是下面这个写法
                # test_set_cate.append([item_cate_dict[item] for item in sess_list[:-1]])
                # 下面是把这个sess里面的cate放到set里面去，只统计出现的cate，顺序就不重要了，再转换成list
                temp_set = set([])
                for item in sess_list[:-1]:
                    temp_set.add(item_cate_dict[item])
                test_set_cate.append(list(temp_set))
                test_set_label.append(sess_list[-1])
                test_set_user.append(user_item)

    # for index, item in enumerate(train_set_input):
    #     print("=====")
    #     print(item)
    #     print(train_set_cate[index])


    pickle.dump((train_set_input, train_set_label), open('train.txt', 'wb'))
    pickle.dump((test_set_input, test_set_label), open('test.txt', 'wb'))

def get_train_test():
    # print("begin get_dict")
    user_sess_dict = get_dict()

    # print("after get_dict")

    # print("begin build_train_set")
    input, user_id, all_user_count, all_item_count, all_cate_count, item_cate_dict = \
        build_train_set_txt(user_sess_dict)
    print("用户/物品/种类数量")
    print(all_user_count)
    print(all_item_count)
    print(all_cate_count)
    pickle.dump((all_user_count, all_item_count, all_cate_count), open('statistic.txt', 'wb'))
    # print("after build_train_set")

    # print("begin split_validation")
    split_validation(input, user_id, item_cate_dict, valid_portion=0.2)
    # print("after split_validation")

get_train_test()