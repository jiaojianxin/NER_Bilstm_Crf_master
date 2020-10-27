# -*- coding:utf-8 -*-
import json

import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as kr


def read_json(read_path):
    with open(read_path, 'r', encoding='utf-8') as f:
        load_dict = json.load(f)
    return load_dict


def read_data(filename):
    """
    :param filename:
    :return: 将文字与标签分开，存入列表
    """
    content, label, sentences, tag = [], [], [], []
    with open(filename, encoding='utf-8') as file:
        lines = file.readlines()
    for eachline in lines:
        if eachline != '\n':
            [char, tag_] = eachline.strip().split()
            sentences.append(char)
            tag.append(tag_)
        else:
            content.append(sentences)
            label.append(tag)
            sentences, tag = [], []
    return content, label


def sequence2id(filename, word2id_path, label2id_path):
    """
    :param filename:
    :return: 将文字与标签，转换为数字
    """
    content2id, label2id = [], []
    content, label = read_data(filename)
    word = read_json(word2id_path)
    tag2label = read_json(label2id_path)
    for i in range(len(label)):
        label2id.append([tag2label[x] for x in label[i]])
    for j in range(len(content)):
        w = []
        for key in content[j]:
            if key[0].isdigit():
                key = '<NUM>'
            if key not in word:
                key = '<UNK>'
            w.append(word[key])
        content2id.append(w)
    return content2id, label2id


def batch_iter(x, y, batch_size=64):
    """
    :param x:
    :param y:
    :param batch_size: 每次进入模型的句子数量
    :return:
    """
    data_len = len(x)
    x = np.array(x)
    y = np.array(y)
    num_batch = int((data_len - 1) / batch_size) + 1  # 计算一个epoch,需要多少次batch

    indices = np.random.permutation(data_len)  # 生成随机数列
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = batch_size * i
        end_id = min(batch_size * (i + 1), data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def process_seq(x_batch):
    """
    :param x_batch: 计算一个batch里面最长句子 长度n
    :param y_batch:动态RNN 保持同一个batch里句子长度一致即可，sequence为实际句子长度
    :return: 对所有句子进行padding,长度为n
    """
    seq_len = []
    max_len = max(map(lambda x: len(x), x_batch))  # 计算一个batch中最长长度
    for i in range(len(x_batch)):
        seq_len.append(len(x_batch[i]))

    x_pad = kr.preprocessing.sequence.pad_sequences(x_batch, max_len, padding='post', truncating='post')
    # y_pad = kr.preprocessing.sequence.pad_sequences(y_batch, max_len, padding='post', truncating='post')

    return x_pad, seq_len


def merge_num(text_list):
    """将数字合并"""
    word_list = list()
    for x in text_list:
        if word_list and (x.isdigit() or x == '.') and word_list[-1][0].isdigit():
            word_list[-1] = word_list[-1] + x
        else:
            word_list.append(x)
    return word_list


def get_bert_vec(filename):
    """读取bert向量"""
    vec_np_list = np.load(filename)
    vec_np_list = vec_np_list.astype('float32')
    vec_tf = tf.constant(vec_np_list)

    return vec_tf
