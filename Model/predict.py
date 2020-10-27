# -*- coding: utf-8 -*-
import tensorflow as tf

from Parameters import Parameters as pm
from Bilstm_Crf import LSTM_CRF
from Public_fuc import read_json, merge_num


def delete_char(text_str):
    """数字处理23,4/23，4/23．4等情况"""
    _res = list()
    _char = [',', '，']
    text_str = text_str.replace('．', '.')
    for i in range(1, len(text_str) - 1):
        x = text_str[i]
        y = text_str[i - 1]
        z = text_str[i + 1]
        if x in _char and y.isdigit() and z.isdigit():
            continue
        else:
            _res.append(x)
    res = text_str[0] + ''.join(_res) + text_str[-1]
    return res


def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    content = delete_char(content)  # 处理23,4/23，4/23．4等情况
    content_list = content.split('\n')

    word_list = [merge_num(list(x)) for x in content_list if x]

    return word_list


def sequence2id(filename, word2id_path):
    """
    :param filename:
    :return: 将文字，转换为数字
    """
    content2id = []
    content = read_file(filename)
    word2id = read_json(word2id_path)
    for j in range(len(content)):
        w = []
        for key in content[j]:
            if key[0].isdigit():
                key = '<NUM>'
            if key not in word2id:
                key = '<UNK>'
            w.append(word2id[key])
        content2id.append(w)
    return content, content2id


def convert(contents, label_line, tag2label):
    label_dict = {v: k for k, v in tag2label.items()}
    _label = [label_dict[i] for i in label_line]
    res_list = list(zip(contents, _label))
    res_list = [' '.join(x) for x in res_list]
    res_str = '\n'.join(res_list)

    return res_str


def val():
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    # save_path = tf.train.latest_checkpoint('./checkpoints/biLstm_crf')
    save_path = tf.train.latest_checkpoint(pm.save_dir)  # latest_checkpoint() 方法查找最新保存的模型
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)

    contents, content2id = sequence2id(pm.eva_data_pre_path, pm.word2id_path)
    label = model.predict(session, content2id)
    return contents, label


if __name__ == '__main__':
    pm = pm
    model = LSTM_CRF()

    contents, label = val()  # 预测标签

    tag2label = read_json(pm.label2id_path)
    res_list = [convert(contents[i], label[i], tag2label) for i in range(len(contents))]
    res_all = '\n'.join(res_list)
    with open(pm.pre_res_path, 'w', encoding='utf-8') as f:
        f.write(res_all)
