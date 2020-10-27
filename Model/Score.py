# -*-coding: utf8 -*-

import numpy as np
import pandas as pd

from collections import Counter

from Parameters import Parameters as pm


def get_label_data(test_dir, pre_dir):
    """
    模型真实和预测标签文件，根据文件读取真实和预测标签，分别存入列表
    将真实的标签和预测的标签转化为符合模型评估入参的模式
    文件内格式：每行包含两列，第一列为真实标签，第二列为预测标签，两列必须以空格为分隔符；
    :param test_dir: 数据文件路径
    :param pre_dir: 数据文件路径
    :return: 返回两个标签列表，例如：['true', 'false',...,'true']
    """
    with open(test_dir, 'r', encoding='utf-8') as f:
        real_list = f.read().replace('\n\n', '\n')[: -1].split('\n')
    with open(pre_dir, 'r', encoding='utf-8') as f:
        pre_list = f.read().split('\n')
    real_label = [x.split(' ')[1] for x in real_list]
    pre_label = [x.split(' ')[1] for x in pre_list]
    real_word = [x.split(' ')[0] for x in real_list if x]
    pre_word = [x.split(' ')[0] for x in pre_list if x]
    for i in range(len(real_word)):
        if real_word[i] != pre_word[i]:
            print(f'{i}    {real_word[i]}   {pre_word[i]}')
            raise ValueError('真实数据与预测数据不对应，请检查！')
    lenth = len(real_label)
    true_labels, predict_labels = list(), list()
    _ = [[true_labels.append(real_label[i]), predict_labels.append(pre_label[i])] for i in range(lenth)]

    return (true_labels, predict_labels)


def matrix_func(true_list, pre_list, labels2idx, confMatrix):
    """
    辅助 create_confusion_matrix 函数 type='2'的情况，
    减少代码冗余度，
    根据一组真实标签与预测出来的标签进行对比，进而生成填充混淆矩阵，
    预测正确率达到3/5就断预测准确，如果达不到，则认为预测标签中出现最多的标签为误判标签；
    :param true_list: 一组真实的标签；
    :param pre_list: 与真实标签照应且等长的预测标签；
    :param labels2idx: 标签字典；
    :param confMatrix: 混淆矩阵
    :return: 更改后的混淆矩阵
    """
    true_label = [x.split('-')[-1] for x in true_list]
    true_idx = labels2idx[true_label[-1]]  # 真实标签所在的位置
    if true_list == pre_list:  # 预测标签中，预测正确率达到3/5就断预测准确
        confMatrix[true_idx][true_idx] += 1
    else:
        pre_label = [x.split('-')[-1] for x in pre_list if x.split('-')[-1] != true_label[-1]]
        if pre_label:
            c = Counter(pre_label)
            pre = c.most_common(1)[0][0]  # 查找预测中最多的标签，为了填补混淆矩阵；
        else:
            pre = 'O'
        pre_idx = labels2idx[pre]  # 预测标签所在的位置
        confMatrix[true_idx][pre_idx] += 1
    return confMatrix


def create_confusion_matrix(test_dir, pre_dir, type='1'):
    """
    创建混淆矩阵，分为多种类型：
    '1' --> 普通多分类问题，标签预测正确或者预测为其他标签；
    '2' --> 数据像BIOES标注问题，标签错位现象；
    '3' --> 待定；
    :param test_dir: 真实的标签数据地址
    :param pre_dir: 预测的标签数据地址
    :param type: 计算的类型，str
    :return: 返回混淆矩阵及标签id字典
    """
    true_labels, predict_labels = get_label_data(test_dir, pre_dir)
    len_true_labels = len(true_labels)
    if type == '1':
        all_labels = {'B-TIME': 0, 'I-TIME': 1, 'E-TIME': 2, 'B-TIMEPRO': 3, 'I-TIMEPRO': 4, 'E-TIMEPRO': 5,
                      'B-TIMETARPRO': 6, 'I-TIMETARPRO': 7, 'E-TIMETARPRO': 8, 'B-AREA': 9, 'I-AREA': 10, 'E-AREA': 11, 'B-AREAPRO': 12,
                      'I-AREAPRO': 13, 'E-AREAPRO': 14, 'B-ADJ': 15, 'I-ADJ': 16, 'E-ADJ': 17, 'B-TAR': 18, 'I-TAR': 19,
                      'E-TAR': 20, 'B-TARPRO': 21, 'I-TARPRO': 22, 'E-TARPRO': 23, 'B-VAL': 24, 'I-VAL': 25, 'E-VAL': 26,
                      'B-UNIT': 27, 'I-UNIT': 28, 'E-UNIT': 29, 'O': 30}
        confMatrix = np.zeros([len(all_labels), len(all_labels)], dtype=np.int32)
        for i in range(len_true_labels):
            true_idx = all_labels[true_labels[i]]
            pre_idx = all_labels[predict_labels[i]]
            confMatrix[true_idx][pre_idx] += 1
        return (confMatrix, all_labels)
    elif type == '2':
        labels2idx = {'TIME': 0, 'TIMEPRO': 1, 'TIMETARPRO': 2, 'AREA': 3, 'AREAPRO': 4, 'ADJ': 5, 'TAR': 6,
                        'TARPRO': 7, 'VAL': 8, 'UNIT': 9, 'O': 10}
        confMatrix = np.zeros([len(labels2idx), len(labels2idx)], dtype=np.int32)
        true_list = list()  # 存放真实标签前几个相同标签
        for label in true_labels:
            if not true_list or true_list[-1].split('-')[-1] == label.split('-')[-1]:  # or，只要前面不满足才会往后判断所以不会出错
                true_list.append(label)
                # 处理最后一组标签
                if len(predict_labels) == len(true_list):
                    confMatrix = matrix_func(true_list, predict_labels, labels2idx, confMatrix)  # 详情见matrix_func
            else:
                pre_list = predict_labels[:len(true_list)]  # 截断与真实标签相等的个数，判断个数
                confMatrix = matrix_func(true_list, pre_list, labels2idx, confMatrix)  # 详情见matrix_func
                predict_labels = predict_labels[len(true_list):]  # 预测标签截断
                true_list = [label]  # 存储新标签

        return (confMatrix, labels2idx)


def calculate_all_prediction(confMatrix, total_sum):
    """
    计算总精度：对角线上所有值除以总数
    :param confMatrix: 混淆矩阵
    :return: 总精度
    """
    correct_sum = (np.diag(confMatrix)).sum()
    prediction = round(100 * float(correct_sum) / float(total_sum), 2)
    return prediction


def calculate_label_prediction(confMatrix, labelidx):
    """
    计算某一个类标预测精度：该类被预测正确的数除以该类的总数
    :param confMatrix: 混淆矩阵
    :param labelidx: 标签索引
    :return: 标签的预测精度
    """
    label_total_sum = confMatrix.sum(axis=0)[labelidx]
    label_correct_sum = confMatrix[labelidx][labelidx]
    prediction = 0
    if label_total_sum != 0:
        prediction = round(100 * float(label_correct_sum) / float(label_total_sum), 2)
    return prediction


def calculate_label_recall(confMatrix, labelidx):
    """
    计算某一个类标的召回率
    :param confMatrix: 混淆矩阵
    :param labelidx: 标签索引
    :return: 标签的召回率
    """
    label_total_sum = confMatrix.sum(axis=1)[labelidx]
    label_correct_sum = confMatrix[labelidx][labelidx]
    recall = 0
    if label_total_sum != 0:
        recall = round(100 * float(label_correct_sum) / float(label_total_sum), 2)
    return recall


def calculate_f1(prediction, recall):
    """
    计算F1值
    :param prediction: 准确率
    :param recall: 召回率
    :return: F1值
    """
    if (prediction + recall) == 0:
        return 0
    return round(2 * prediction * recall / (prediction + recall), 2)


def get_parameter(test_dir, pre_dir):
    """
    将真实的标签和预测的标签转化为符合模型评估入参的模式
    :param test_dir: 真实标签 两列 --> 字   标签
    :param pre_dir: 预测标签 两列 --> 字   标签
    :return: 模型Score评估的入参，两列 --> 真实标签列表，预测标签列表
    """
    with open(test_dir, 'r', encoding='utf-8') as f:
        real_list = f.read().replace('\n\n', '\n')[: -1].split('\n')
    with open(pre_dir, 'r', encoding='utf-8') as f:
        pre_list = f.read().split('\n')
    real_label = [x.split(' ')[1] for x in real_list]
    pre_label = [x.split(' ')[1] for x in pre_list]
    real_word = [x.split(' ')[0] for x in real_list if x]
    pre_word = [x.split(' ')[0] for x in pre_list if x]
    for i in range(len(real_word)):
        if real_word[i] != pre_word[i]:
            print(f'{i}    {real_word[i]}   {pre_word[i]}')
            raise ValueError('真实数据与预测数据不对应，请检查！')
    lenth = len(real_label)
    true_labels, predict_labels = list(), list()
    _ = [[true_labels.append(real_label[i]), predict_labels.append(pre_label[i])] for i in range(lenth)]

    return (true_labels, predict_labels)


def confusion_matrix_score(test_dir, pre_dir, type='1', save_res_dir=''):
    """
    主函数，控制计算类型和结果的保存，计算精度、召回率、F1值
    :param file_dir: 见get_label_data参数说明
    :param type: 见create_confusion_matrix参数说明
    :param save_res_dir: 结果默认打印，如果输入地址，结果将保存此地址，且保存为csv格式
    :return:
    """
    if type not in ['1', '2']:
        raise ValueError('请根据需要输入正确 type 值')
    if save_res_dir:
        if save_res_dir.split('.')[-1] != 'csv':
            raise ValueError('保存文件为csv文件，请检查保存地址是否符合规定！')
    res_list = list()  # 用于保存结果
    # 获取文件并返回混淆矩阵和标签字典
    confMatrix, label2idx = create_confusion_matrix(test_dir, pre_dir, type=type)
    total_sum = int(confMatrix.sum())  # 数据总量
    all_prediction = calculate_all_prediction(confMatrix, total_sum)
    res_list.append([f'数据总数：{total_sum}'])
    res_list.append([f'标签数量：{len(label2idx)}'])
    id_label = {y: x for x, y in label2idx.items()}
    label_list = ['']
    a = [label_list.append(id_label[i]) for i in range(len(label2idx))]
    res_list.append([''])
    res_list.append(label_list)
    label_prediction = []
    label_recall = []

    for i in range(len(label2idx)):
        _res_list = list()
        label_prediction.append(calculate_label_prediction(confMatrix, i))
        label_recall.append(calculate_label_recall(confMatrix, i))
        _res_list.append(id_label[i])
        for j in range(len(label2idx)):
            _res_list.append(confMatrix[i][j])
        res_list.append(_res_list)
    res_list.append([''])
    res_list.append([f'总的准确率为：{all_prediction}%'])
    res_list.append([''])
    res_list.append(['', 'prediction', 'recall', 'F1'])
    for i in range(len(label2idx)):
        res_list.append([id_label[i], f'{label_prediction[i]}%', f'{label_recall[i]}%',
                         f'{calculate_f1(label_prediction[i], label_recall[i])}%'])

    p = round(np.array(label_prediction).sum() / len(label_prediction), 2)
    r = round(np.array(label_recall).sum() / len(label_prediction), 2)
    res_list.append([''])
    res_list.append([f'Averaged-prediction: {p}, recall: {r}, F1: {calculate_f1(p, r)}'])
    max_len = max([len(x) for x in res_list])
    res_list = [x + [''] * (max_len - len(x)) for x in res_list]

    res = pd.DataFrame(res_list)
    if save_res_dir:
        res.to_csv(save_res_dir, index=False, header=False)
    print(res)
    return res


if __name__ == '__main__':
    eva_data_real_path = pm.eva_data_real_path  # 测试数据
    pre_res_path = pm.pre_res_path  # 预测结果数据
    score_path = pm.score_path3  # 评估类型1的结果
    score_path2 = pm.score_path4  # 评估类型2的结果
    res = confusion_matrix_score(eva_data_real_path, pre_res_path, type='1', save_res_dir=score_path)
    res2 = confusion_matrix_score(eva_data_real_path, pre_res_path, type='2', save_res_dir=score_path2)
