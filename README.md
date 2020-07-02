# BIT_DM_PROJECT
**所需库**
- torch
- pandas
- numpy
- pickle

**代码执行顺序**
1. 将[原始数据集](https://tianchi.aliyun.com/dataset/dataDetail?dataId=42 "原始数据集")中的user_log_format1.csv放入dataset目录下（dataset中有样例文件）；
2. 依次执行dataset中的split_session_tmall_v2.py，view_df.py，build_dataset.py来构建训练集和测试集；
3. 执行main.py进行训练和测试。