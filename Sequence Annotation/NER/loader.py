# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from transformers import BertTokenizer

"""
数据加载模块
"""

class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config  # 配置参数
        self.path = data_path  # 数据路径
        self.tokenizer = load_vocab(config["bert_path"])  # 加载BERT分词器
        self.sentences = []  # 存储句子
        self.schema = self.load_schema(config["schema_path"])  # 加载标签模式
        self.load()  # 加载数据

    def load(self):
        self.data = []  # 存储数据
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")  # 按段落分隔
            for segment in segments:
                sentenece = []  # 存储单个句子
                labels = [8]  # 初始标签（cls_token）
                for line in segment.split("\n"):
                    if line.strip() == "":  # 跳过空行
                        continue
                    char, label = line.split()  # 分割字符和标签
                    sentenece.append(char)  # 存储字符
                    labels.append(self.schema[label])  # 存储标签对应的索引
                sentence = "".join(sentenece)  # 拼接句子
                self.sentences.append(sentence)  # 添加句子到列表
                input_ids = self.encode_sentence(sentenece)  # 编码句子
                labels = self.padding(labels, -1)  # 填充标签
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])  # 存储处理后的数据
        return

    def encode_sentence(self, text, padding=True):
        # 使用BERT分词器编码文本，并进行填充或截断
        return self.tokenizer.encode(text, 
                                     padding="max_length",
                                     max_length=self.config["max_length"],
                                     truncation=True)

    def decode(self, sentence, labels):
        # 解码标签并输出结果，方便调试
        sentence = "$" + sentence
        labels = "".join([str(x) for x in labels[:len(sentence)+2]])
        results = defaultdict(list)
        for location in re.finditer("(04+)", labels):
            s, e = location.span()
            print("location", s, e)
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            print("org", s, e)
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            print("per", s, e)
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            print("time", s, e)
            results["TIME"].append(sentence[s:e])
        return results

    def padding(self, input_id, pad_token=0):
        # 填充或截断输入序列，使其长度一致
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        # 返回数据集的长度
        return len(self.data)

    def __getitem__(self, index):
        # 获取特定索引的数据项
        return self.data[index]

    def load_schema(self, path):
        # 加载标签模式文件
        with open(path, encoding="utf8") as f:
            return json.load(f)

def load_vocab(vocab_path):
    # 加载BERT分词器的词汇表
    return BertTokenizer.from_pretrained(vocab_path)

# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("ner_data/train", Config)
    dl = DataLoader(dg, batch_size=32)
    for x, y in dl:
        print(x.shape, y.shape)
        print(x[1], y[1])
        input()
