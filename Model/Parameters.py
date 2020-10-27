# -*- coding:utf-8 -*-
class Parameters(object):
    """模型参数"""
    num_epochs = 60  # 迭代次数
    # embedding_size = 100  # 向量维度
    embedding_size = 768  # 向量维度
    batch_size = 64  # 每批进入模型句数，CPU对2的指数不友好，GPU用2的指数较好
    hidden_dim = 128  # 神经网络层数
    learning_rate = 0.001  # 学习率
    clip = 4.0  # 梯度剪裁，控制参数调节的大小，防止梯度爆炸
    lr = 0.95  # 学习率下降速度
    keep_pro = 1.0  # 向量防止过拟合正则化（丢失）程度
    num_tags = 34  # 标签个数

    train_data_path = r'../Data/label_data/train_data'  # 训练数据
    test_data_path = r'../Data/label_data/test_data'  # 测试数据
    eva_data_real_path = r'../Data/label_data/eva_data_real'   # 验证数据（打标）
    eva_data_pre_path = r'../Data/label_data/eva_data_pre'   # 验证数据（未打标）
    pre_res_path = r'../Data/res_data/predict_res_data'  # 预测结果数据
    score_path1 = r'../Data/res_data/score_3.csv'  # 评估类型1的结果(精确匹配)
    score_path2 = r'../Data/res_data/score_4.csv'  # 评估类型2的结果(精确匹配)

    word_vec_path = r'../Data/label_data/vec.npy'  # 向量
    word2id_path = r'../Data/label_data/word2id.json'  # 字ID
    label2id_path = r'../Data/label_data/label2id.json'  # 标签ID

    tensorboard_dir = r'./tensorboard'  # tensorflow计算图保存地址
    save_dir = r'./checkpoints'  # 模型保存地址
