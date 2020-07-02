import pandas as pd
import codecs
import pickle
from tqdm import tqdm

def get_user_sess_from_df(df):

    # print("begin")
    # group = df.groupby(['user_id'])
    # print(len(group))
    #
    # print("begin")
    # group = df.groupby(['item_id'])
    # print(len(group))

    fp = open("paper_all_dict.txt", "rb+")
    (user_dict, item_dict, cate_dict, item_cate_dict) = pickle.load(fp)

    df = df.sort_values(by='time_stamp')

    # print(df)

    group = df.groupby(['user_id', 'time_stamp'])

    # print(group.count())

    user_sess_dict = dict()

    for item in tqdm(group):
        user_id = int(item[0][0])

        user_id = user_dict[user_id]

        group_df = item[1]

        sess = []
        for index, row in group_df.iterrows():
            sess.append(item_dict[int(row['item_id'])])

        # print(sess)

        try:
            user_sess_dict[user_id].append(sess)
        except:
            user_sess_dict[user_id] = [sess]

    pickle.dump(user_sess_dict, open('paper_user_sess_dict.txt', 'wb'))


def generate_dict(df):

    user_count = 0
    item_count = 1
    cate_count = 1
    user_dict = dict()
    item_dict = dict()
    cate_dict = dict()
    # 先把用来填充的dict补上，方便后面为item和cate建立embedding层
    item_dict[-1] = 0
    cate_dict[-1] = 0
    item_cate_dict = dict()
    cur_item = 0
    cur_cate = 0
    for index, row in tqdm(df.iterrows()):
        user_id = int(row['user_id'])
        item_id = int(row['item_id'])
        cate_id = int(row['cat_id'])
        if user_id not in user_dict:
            user_dict[user_id] = user_count
            user_count += 1
        if item_id not in item_dict:
            item_dict[item_id] = item_count
            item_count += 1
        cur_item = item_dict[item_id]
        if cate_id not in cate_dict:
            cate_dict[cate_id] = cate_count
            cate_count += 1
        cur_cate = cate_dict[cate_id]
        if cur_item not in item_cate_dict:
            item_cate_dict[cur_item] = cur_cate

    pickle.dump((user_dict, item_dict, cate_dict, item_cate_dict), open('paper_all_dict.txt', 'wb'))

# read_df()

print("read_df")
df = pd.read_csv('result.csv')
# df = df[:1000]

print("generate")
generate_dict(df)

print("get")
get_user_sess_from_df(df)