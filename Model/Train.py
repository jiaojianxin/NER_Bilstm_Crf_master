import os

from Bilstm_Crf import LSTM_CRF
from Parameters import Parameters as pm
from Public_fuc import sequence2id, process_seq, batch_iter

import tensorflow as tf


def train():
    if not os.path.exists(pm.tensorboard_dir):
        os.makedirs(pm.tensorboard_dir)
    if not os.path.exists(pm.save_dir):
        os.makedirs(pm.save_dir)
    save_path = os.path.join(pm.save_dir, 'best_validation')

    tf.summary.scalar('loss', model.loss)  # 显示量标信息 loss
    merged_summary = tf.summary.merge_all()  # 显示训练时的各种信息
    writer = tf.summary.FileWriter(pm.tensorboard_dir)  # 指定一个文件保存图
    saver = tf.train.Saver()  # 保存模型的变量
    session = tf.Session()  # 控制会话
    session.run(tf.global_variables_initializer())  # tf启动
    writer.add_graph(session.graph)  # 可视化图表
    # 获取训练和测试的字和标签
    content_train, label_train = sequence2id(pm.train_data_path, pm.word2id_path, pm.label2id_path)
    content_test, label_test = sequence2id(pm.test_data_path, pm.word2id_path, pm.label2id_path)
    loss_list = [[100.0, 100.0]]  # 用于保存损失率大小有助于保存模型时使用；
    for epoch in range(pm.num_epochs):
        print('Epoch:', epoch + 1)
        # num_batchs = int((len(content_train) - 1) / pm.batch_size) + 1
        batch_train = batch_iter(content_train, label_train, batch_size=pm.batch_size)
        for x_batch, y_batch in batch_train:
            x_batch, seq_leng_x = process_seq(x_batch)
            y_batch, seq_leng_y = process_seq(y_batch)
            feed_dict = model.feed_data(x_batch, y_batch, seq_leng_x, pm.keep_pro)
            _, global_step, loss, tain_summary = session.run(
                [model.optimizer, model.global_step, model.loss, merged_summary],
                feed_dict=feed_dict)
            if global_step % 100 == 0:
                test_loss = model.test(session, content_test, label_test)
                print('global_step:', global_step, 'train_loss:', loss, 'test_loss:', test_loss)
                loss_train_test = [float(loss), float(test_loss)]
                if loss_list and sum(loss_train_test) < sum(loss_list[-1]):
                    print('Saving Model...')
                    saver.save(session, save_path=save_path, global_step=global_step)
                    loss_list.append(loss_train_test)
                if len(loss_list) > 5:
                    del loss_list[0]
                    
        # 学习率梯度下降
        pm.learning_rate *= pm.lr
    train_test_loss = '\n'.join([' <----> '.join([str(x), str(y)]) for x,y in loss_list])
    print(train_test_loss)
    with open(r'../Data/res_data/train_test_loss.txt', 'w', encoding='utf-8') as f:
        f.write(train_test_loss)


if __name__ == '__main__':
    pm = pm
    model = LSTM_CRF()
    train()
