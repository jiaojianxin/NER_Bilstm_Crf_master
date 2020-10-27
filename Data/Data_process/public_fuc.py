# -*- coding: utf8 -*-
import json


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


def merge_num(word_label, text_list):
    """将数字合并"""
    # if len(word_label.split(' ')) == 1:
    #     print(word_label)
    word, label = word_label.split(' ')
    if len(text_list) != 0 and (word.isdigit() or word == '.') and text_list[-1][0][0].isdigit():
        text_list[-1][0] = text_list[-1][0] + word
    else:
        text_list.append([word, label])


def write_json(data_dict, write_path):
    with open(write_path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False)


def read_json(read_path):
    with open(read_path, 'r', encoding='utf-8') as f:
        load_dict = json.load(f)
    return load_dict
