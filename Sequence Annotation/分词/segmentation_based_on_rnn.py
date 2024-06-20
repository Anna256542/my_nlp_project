#coding:utf8

import torch
import torch.nn as nn
import jieba
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset

"""
基于pytorch的网络编写一个分词模型
我们使用jieba分词的结果作为训练数据
看看是否可以得到一个效果接近的神经网络模型
"""

class TorchModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_rnn_layers, vocab):
        super(TorchModel, self).__init__()
        # 词嵌入层，将词汇表中的词映射到一个定长的向量
        self.embedding = nn.Embedding(len(vocab) + 1, input_dim)  # shape=(vocab_size, dim)
        # RNN层，进行序列处理
        self.rnn_layer = nn.RNN(input_size=input_dim,
                                hidden_size=hidden_size,
                                batch_first=True,
                                bidirectional=False,
                                num_layers=num_rnn_layers,
                                nonlinearity="relu",
                                dropout=0.1)
        # 分类层，将RNN的输出映射到分类标签
        self.classify = nn.Linear(hidden_size, 2)
        # 损失函数，用于计算分类任务的损失
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, x, y=None):
        # 前向传播过程
        x = self.embedding(x)  # output shape:(batch_size, sen_len, input_dim)
        x, _ = self.rnn_layer(x)  # output shape:(batch_size, sen_len, hidden_size)
        y_pred = self.classify(x)  # input shape:(batch_size, sen_len, class_num)
        # 如果提供了标签y，计算并返回损失值；否则返回预测结果
        if y is not None:
            return self.loss_func(y_pred.view(-1, 2), y.view(-1))
        else:
            return y_pred

class TextDataset(Dataset):
    def __init__(self, corpus_path, vocab, max_length):
        self.vocab = vocab
        self.corpus_path = corpus_path
        self.max_length = max_length
        self.data = self.load_data()

    def load_data(self):
        # 加载语料库数据并进行预处理
        data = []
        with open(self.corpus_path, encoding="utf8") as f:
            for line in f:
                sequence = sentence_to_sequence(line.strip(), self.vocab)
                label = sequence_to_label(line.strip())
                sequence, label = self.padding(sequence, label)
                data.append([torch.LongTensor(sequence), torch.LongTensor(label)])
                if len(data) > 10000:  # 限制数据量
                    break
        return data

    def padding(self, sequence, label):
        # 对序列和标签进行填充，使其达到最大长度
        sequence = sequence[:self.max_length] + [0] * (self.max_length - len(sequence))
        label = label[:self.max_length] + [-100] * (self.max_length - len(label))
        return sequence, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

def sentence_to_sequence(sentence, vocab):
    # 将句子转换为词汇表对应的索引序列
    return [vocab.get(char, vocab['unk']) for char in sentence]

def sequence_to_label(sentence):
    # 根据分词结果生成标签序列，0表示词中间，1表示词的结尾
    words = jieba.lcut(sentence)
    label = [0] * len(sentence)
    pointer = 0
    for word in words:
        pointer += len(word)
        label[pointer - 1] = 1
    return label

def build_vocab(vocab_path):
    # 构建词汇表，将字符映射到索引
    vocab = {}
    with open(vocab_path, "r", encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line.strip()
            vocab[char] = index + 1
    vocab['unk'] = len(vocab) + 1  # 未知字符的索引
    return vocab

def build_dataset(corpus_path, vocab, max_length, batch_size):
    # 构建数据集，并返回数据加载器
    dataset = TextDataset(corpus_path, vocab, max_length)
    return DataLoader(dataset, shuffle=True, batch_size=batch_size)

def train_model(epoch_num, batch_size, char_dim, hidden_size, num_rnn_layers, max_length, learning_rate, vocab_path, corpus_path):
    # 模型训练过程
    vocab = build_vocab(vocab_path)
    data_loader = build_dataset(corpus_path, vocab, max_length, batch_size)
    model = TorchModel(char_dim, hidden_size, num_rnn_layers, vocab)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, y in data_loader:
            optimizer.zero_grad()
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
        print(f"=========\n第{epoch + 1}轮平均loss: {np.mean(watch_loss)}")

    torch.save(model.state_dict(), "model.pth")
    with open("vocab.json", "w", encoding="utf8") as writer:
        writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))

def predict(model_path, vocab_path, input_strings):
    # 模型预测过程
    char_dim = 50
    hidden_size = 100
    num_rnn_layers = 3
    vocab = build_vocab(vocab_path)
    model = TorchModel(char_dim, hidden_size, num_rnn_layers, vocab)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for input_string in input_strings:
        x = sentence_to_sequence(input_string, vocab)
        with torch.no_grad():
            result = model(torch.LongTensor([x]))[0]
            result = torch.argmax(result, dim=-1).numpy()
            segmented_sentence = ''.join([char if tag == 0 else char + ' ' for char, tag in zip(input_string, result)])
            print(segmented_sentence)

if __name__ == "__main__":
    # 训练模型
    train_model(
        epoch_num=10,
        batch_size=20,
        char_dim=50,
        hidden_size=100,
        num_rnn_layers=3,
        max_length=20,
        learning_rate=1e-3,
        vocab_path="chars.txt",
        corpus_path="corpus"
    )

    # 预测示例
    input_strings = [
        "同时国内有望出台新汽车刺激方案",
        "沪胶后市有望延续强势",
        "经过两个交易日的强势调整后",
        "昨日上海天然橡胶期货价格再度大幅上扬"
    ]
    predict("model.pth", "chars.txt", input_strings)
