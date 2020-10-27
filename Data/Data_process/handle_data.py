# -*- coding: utf8 -*-
import os
import numpy as np
from bert_serving.client import BertClient

from public_fuc import delete_char, write_json, read_json, merge_num


class HandleData(object):
    """数据处理"""

    def __init__(self, filename):
        self.labels = ['O', 'TIME', 'TIMEPRO', 'TIMETARPRO', 'TAR', 'TARPRO', 'AREA', 'AREAPRO', 'ADJ', 'VAL', 'UNIT', 'CY']
        self.filename = filename
        self.text_str = self.read()
        self.text_list = self.get_text_list(self.text_str)  # 主要处理数字与点相连的问题，# 如：2.   1.

        self.check_data(self.text_list)

    def read(self):
        with open(self.filename, 'r', encoding='utf-8') as f:
            text_str = f.read()
        text_str = delete_char(text_str)  # 数字部分的处理
        return text_str

    def get_text_list(self, text_str):
        text_str_split = text_str.split('\n\n')
        text_str_split = [x.split('\n') for x in text_str_split if x]

        res_list = []
        for x in text_str_split:
            text_list = []
            _ = [merge_num(y, text_list) for y in x if y]
            res_list.append(text_list)
        res_list = [[' '.join(y) for y in x] for x in res_list]
        return res_list

    def check_data(self, text_list):
        """数据检查，错标、多标、少标等情况"""
        a = 0
        error_num = list()
        for x in text_list:
            for y in x:
                if y and (len(y.split(' ')) != 2 or y.split(' ')[1] not in self.labels):
                    error_num.append(f'{a + 1}  ===  {y}')
                a += 1
        if len(error_num)>0:
            raise ValueError(f'请修改{" , ".join(error_num)}行的错标/多标/少标错误！')
        print('数据检查完毕！')

    def cut_data(self, save_path=None):
        """
        将数据进行BIEO分割
        save_path: 结果写入文本的地址，如果None则不进行写入操作
        """

        # def get_BIEO_data(BIEO_list, label_data):
        #     """
        #     将数据进行BIEO分割
        #     :param res_list: 存放结果的列表
        #     :param label_data: 标签数据--> '河南省 AREA'
        #     :return:
        #     """
        #     word, label = label_data.split(' ')
        #     word_cut = list(word)
        #     word_list = list()
        #     for x in word_cut:
        #         if len(word_list) > 0 and (x.isdigit() or x == '.') and word_list[-1][0].isdigit():
        #             word_list[-1] = word_list[-1] + x
        #         else:
        #             word_list.append(x)
        #
        #     if label == 'O':
        #         BIEO_data = [x + ' ' + 'O' for x in word_list]
        #     else:
        #         if len(word_list) == 1:
        #             BIEO_data = [word_list[0] + ' ' + 'B-' + label]
        #         else:
        #             a = [word_list[0] + ' ' + 'B-' + label]
        #             I_data = word_list[1:-1]
        #             b = [x + ' ' + 'I-' + label for x in I_data]
        #             c = [word_list[-1] + ' ' + 'E-' + label]
        #             BIEO_data = a + b + c
        #     BIEO_list += BIEO_data

        BIEO_list = list()
        for x in self.text_list:
            _BIEO_list = list()
            for label_data in x:
                word, label = label_data.split(' ')
                word_cut = list(word)
                word_list = list()
                for x in word_cut:
                    if len(word_list) > 0 and (x.isdigit() or x == '.') and word_list[-1][0].isdigit():
                        word_list[-1] = word_list[-1] + x
                    else:
                        word_list.append(x)

                if label == 'O':
                    BIEO_data = [x + ' ' + 'O' for x in word_list]
                else:
                    if len(word_list) == 1:
                        BIEO_data = [word_list[0] + ' ' + 'B-' + label]
                    else:
                        a = [word_list[0] + ' ' + 'B-' + label]
                        I_data = word_list[1:-1]
                        b = [x + ' ' + 'I-' + label for x in I_data]
                        c = [word_list[-1] + ' ' + 'E-' + label]
                        BIEO_data = a + b + c
                _BIEO_list += BIEO_data
            BIEO_list.append(_BIEO_list)

        if save_path:
            BIEO_str = '\n\n'.join(['\n'.join(x) for x in BIEO_list])
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(BIEO_str)
        print('BIEO标注完毕！')
        return BIEO_list

    def reduce_sentence(self, BIEO_list_list, num=31, save_path=None):
        """将长句压缩到30字以下"""
        symbol_list = ['。', '，', '；', '！']
        sectence_list = []
        for BIEO_list in BIEO_list_list:
            res_list = [[]]
            word_str = list()  # 存放词语，判断是否大于30
            BIEO_list_len = len(BIEO_list)
            for i in range(BIEO_list_len):
                if i == BIEO_list_len - 1:
                    word_str.append(BIEO_list[i])
                    if len(res_list[-1]) + len(word_str) < num:
                        res_list[-1] += word_str
                    else:
                        res_list.append(word_str)
                    break
                a = BIEO_list[i].split(' ')
                if a[0] and a[0] in symbol_list:
                    word_str.append(BIEO_list[i])
                    if len(res_list[-1]) + len(word_str) < num:
                        res_list[-1] += word_str
                    else:
                        res_list.append(word_str)
                    word_str = list()
                else:
                    word_str.append(BIEO_list[i])
            sectence_list.append(res_list)
        c = [['\n'.join(y) for y in x if y] for x in sectence_list]
        d = ['\n\n'.join(x) for x in c]
        BIEO_res_str = '\n\n'.join(d).replace('\n\n\n', '\n\n') + '\n'
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(BIEO_res_str)
        print('将长句压缩完毕！')
        return BIEO_res_str

    def save_train_data(self, BIEO_res_str, train_path, train_path2):
        with open(train_path2, 'r', encoding='utf-8') as f:
            old_data = f.read()
        all_data = old_data + '\n' + BIEO_res_str
        with open(train_path, 'w', encoding='utf-8') as f:
            f.write(all_data)
        print('训练数据保存成功！')

    def get_new_word(self, BIEO_res_str, word2id_path, vec_path):
        """获取新加入的词，为了增加字典ID及增加向量"""
        word2id_dict = read_json(word2id_path)
        BIEO_list = BIEO_res_str.split('\n')
        word_list = [x.split(' ')[0] for x in BIEO_list if x]
        word_list = [x for x in word_list if not x[0].isdigit() and x != '\n']
        old_word = word2id_dict.keys()
        new_word_list = list(set([x for x in word_list if x not in old_word]))
        if len(new_word_list) > 0:
            word2id_dict_len = len(word2id_dict)
            for i in range(len(new_word_list)):
                word2id_dict[new_word_list[i]] = word2id_dict_len+i

            bc = BertClient('192.168.106.10')
            new_vec_list = np.array([bc.encode([x])[0] for x in new_word_list])
            old_vec_list = np.load(vec_path)
            vec_list = np.concatenate((old_vec_list, new_vec_list))

            np.save(vec_path, vec_list)
            print('字ID： ',len(word2id_dict))
            print('向量个数： ',len(vec_list))
            write_json(word2id_dict, word2id_path)
            print('新数据： ', new_word_list)
            print('bert向量储存成功！')


if __name__ == '__main__':
    filename = r'../label_data/text_label.txt'
    train_path = r'../label_data/train_data'
    train_path2 = r'../label_data/train_data2'
    word2id_path = r'../label_data/word2id.json'
    vec_path = r'../label_data/vec.npy'
    handle_data = HandleData(filename)
    # 进行BIEO标注
    BIEO_list = handle_data.cut_data(save_path=r'../label_data/reduce_BIEO.txt')
    BIEO_res_str = handle_data.reduce_sentence(BIEO_list)
    handle_data.save_train_data(BIEO_res_str, train_path, train_path2)
    handle_data.get_new_word(BIEO_res_str, word2id_path, vec_path)
