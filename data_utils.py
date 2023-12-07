# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, XLNetTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
modelPath = os.path.dirname(os.path.realpath(__file__))


def build_tokenizer(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text = lines[i]
                target = lines[i + 1].strip()
                text_raw = text + " " + target
                text += text_raw + " "

        tokenizer = Tokenizer()
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            try:
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
            except:
                print('WARNING: corrupted word vector of {} when being loaded from GloVe.'.format(tokens[0]))
    return word_vec

def build_embedding_matrix(word2idx, embed_dim, type, w2v_name):
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(embed_dim), type)
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors ...')
        print('word2idx:{}'.format(word2idx))

        embedding_matrix = np.zeros((len(word2idx), embed_dim))  # idx 0 and 1 are all-zeros
        embedding_matrix[1, :] = np.random.uniform(-1 / np.sqrt(embed_dim), 1 / np.sqrt(embed_dim), (1, embed_dim))
        fname = modelPath + '/word-embedding/w2v/'+w2v_name+'/word300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v: k for k, v in word2idx.items()}

    def fit_on_text(self, text):
        text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1


    def text_to_sequence(self, text):

        text = text.strip()
        words = text.split()
        unknownidx = 1
        # print(words)
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        # print(sequence)
        if len(sequence) == 0:
            sequence = [0]
        # print('sequence:{}'.format(len(sequence)))
        return sequence


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):

        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class Tokenizer4XLNet:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = XLNetTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        # fin = open(fname + '.graph.stance.3way11t', 'rb')
        # idx2gragh = pickle.load(fin)
        # fin.close()

        all_data = []
        for i in range(0, len(lines), 3):
            # text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            # text = [s.strip() for s in lines[i].split(" ")]
            # aspect = lines[i + 1].lower().strip()
            text = lines[i].strip()
            target = lines[i + 1].strip()
            stance = lines[i + 2].strip()
            feature = self.statisCount(text)
            seq_len = len(text.split())
            # text = textCount+" "+text
            text_raw_indices = tokenizer.text_to_sequence(text + " " + target)
            text_raw_without_target_indices = tokenizer.text_to_sequence(text)
            target_indices = tokenizer.text_to_sequence(target)
            context_len = np.sum(text_raw_without_target_indices != 0)
            target_len = np.sum(target_indices != 0)
            stance = int(stance) + 1

            text_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text + ' [SEP] ' + target + ' [SEP]')
            # print("'[CLS] ' + text + ' [SEP] ' + target + ' [SEP]':{}".format('[CLS] ' + text + ' [SEP] ' + target + ' [SEP]'))
            # print("text_bert_indices after text_to_sequence:{}".format(text_bert_indices))
            bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (target_len + 1))
            # print("bert_segments_ids:{}".format(bert_segments_ids))
            attention_mask = [1] * len(text_bert_indices)
            attention_mask = np.asarray(attention_mask, dtype='int64')

            bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)

            text_raw_bert_indices = tokenizer.text_to_sequence("[CLS] " + text + " [SEP]")
            target_bert_indices = tokenizer.text_to_sequence("[CLS] " + target + " [SEP]")
            src_mask = [0] + [1] * context_len + [0] * (tokenizer.max_seq_len - context_len - 1)
            src_mask = np.asarray(src_mask[:128], dtype='int64')

            attention_mask_raw = [1] * len(text_raw_bert_indices)
            attention_mask_raw = np.asarray(attention_mask_raw, dtype='int64')
            attention_mask_raw_target = [1] * len(target_bert_indices)
            attention_mask_raw_target = np.asarray(attention_mask_raw_target, dtype='int64')
            # pad adj
            context_asp_adj_matrix = np.zeros(
                (tokenizer.max_seq_len, tokenizer.max_seq_len)).astype('float32')

            # in_graph = idx2gragh[i]
            # if (seq_len <= 128):
            #     context_asp_adj_matrix[:seq_len, :seq_len] = in_graph
            # else :
            #     context_asp_adj_matrix[:128, :128] = in_graph[:128, :128]
            #
            # # print(in_graph)
            # # print(context_asp_adj_matrix)
            # assert in_graph.shape[0] == seq_len, "length error"

            data = {
                'text_bert_indices': text_bert_indices,
                'text_len': seq_len,
                'attention_mask': attention_mask,
                'bert_segments_ids': bert_segments_ids,
                'text_raw_bert_indices': text_raw_bert_indices,
                'attention_mask_raw': attention_mask_raw,
                'target_bert_indices': target_bert_indices,
                'attention_mask_raw_target': attention_mask_raw_target,
                'text_raw_indices': text_raw_indices,
                'text_raw_without_target_indices': text_raw_without_target_indices,
                'target_indices': target_indices,
                'stance': stance,
                'adj_matrix': context_asp_adj_matrix,
                'src_mask': src_mask,

            }

            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def statisCount(self, sentence):
        # statis = []
        # statis.append(sentence.count("#"))
        # statis.append(sentence.count("?"))
        # statis.append(len(sentence))
        strCount = str(len(sentence)) + " " + str(sentence.count("#")) + " " + str(sentence.count("?"))
        return strCount


class predictDataset(Dataset):
    def __init__(self, fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = []
        for i in range(0, len(lines), 3):
            # text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            # text = [s.strip() for s in lines[i].split(" ")]
            # aspect = lines[i + 1].lower().strip()
            text = lines[i]
            target = lines[i + 1].strip()
            stance = lines[i + 2].strip()

            text_raw_indices = tokenizer.text_to_sequence(text + " " + target)
            text_raw_without_target_indices = tokenizer.text_to_sequence(text)
            target_indices = tokenizer.text_to_sequence(target)
            context_len = np.sum(text_raw_without_target_indices != 0)
            target_len = np.sum(target_indices != 0)
            stance = int(stance) + 1

            text_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text + ' [SEP] ' + target + " [SEP]")
            bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (target_len + 1))
            bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)

            text_raw_bert_indices = tokenizer.text_to_sequence("[CLS] " + text + " [SEP]")
            target_bert_indices = tokenizer.text_to_sequence("[CLS] " + target + " [SEP]")

            # text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            # text_raw_without_aspect_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            # text_left_indices = tokenizer.text_to_sequence(text_left)
            # text_left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
            # text_right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
            # text_right_with_aspect_indices = tokenizer.text_to_sequence(" " + aspect + " " + text_right, reverse=True)
            # aspect_indices = tokenizer.text_to_sequence(aspect)
            # left_context_len = np.sum(text_left_indices != 0)
            # aspect_len = np.sum(aspect_indices != 0)
            # aspect_in_text = torch.tensor([left_context_len.item(), (left_context_len + aspect_len - 1).item()])  # may come back later
            # polarity = int(polarity) + 1


            data = {
                'text_bert_indices': text_bert_indices,
                'bert_segments_ids': bert_segments_ids,
                'text_raw_bert_indices': text_raw_bert_indices,
                'target_bert_indices': target_bert_indices,
                'text_raw_indices': text_raw_indices,
                'text_raw_without_target_indices': text_raw_without_target_indices,
                'target_indices': target_indices,
                'stance': stance,
            }

            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class XLNetDataset(Dataset):
    def __init__(self, fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = []
        for i in range(0, len(lines), 3):
            text = lines[i]
            target = lines[i + 1].strip()
            stance = lines[i + 2].strip()

            text_raw_indices = tokenizer.text_to_sequence(text + " " + target)
            text_raw_without_target_indices = tokenizer.text_to_sequence(text)
            target_indices = tokenizer.text_to_sequence(target)
            context_len = np.sum(text_raw_without_target_indices != 0)
            target_len = np.sum(target_indices != 0)
            stance = int(stance) + 1

            text_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text + ' [SEP] ' + target + ' [SEP]')
            # print("'[CLS] ' + text + ' [SEP] ' + target + ' [SEP]':{}".format('[CLS] ' + text + ' [SEP] ' + target + ' [SEP]'))
            # print("text_bert_indices after text_to_sequence:{}".format(text_bert_indices))
            bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (target_len + 1))
            # print("bert_segments_ids:{}".format(bert_segments_ids))

            bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)

            text_raw_bert_indices = tokenizer.text_to_sequence("[CLS] " + text + " [SEP]")
            target_bert_indices = tokenizer.text_to_sequence("[CLS] " + target + " [SEP]")

            input_ids = tokenizer.text_to_sequence(text + '[SEP]' + target + ' [SEP] [CLS]')

            attention_mask = [float(i > 0) for i in input_ids]

            data = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'text_bert_indices': text_bert_indices,
                'bert_segments_ids': bert_segments_ids,
                'text_raw_bert_indices': text_raw_bert_indices,
                'target_bert_indices': target_bert_indices,
                'text_raw_indices': text_raw_indices,
                'text_raw_without_target_indices': text_raw_without_target_indices,
                'target_indices': target_indices,
                'stance': stance,

            }

            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def statisCount(self, sentence):
        # statis = []
        # statis.append(sentence.count("#"))
        # statis.append(sentence.count("?"))
        # statis.append(len(sentence))
        strCount = str(len(sentence)) + " " + str(sentence.count("#")) + " " + str(sentence.count("?"))
        return strCount


class LSTMDataset(Dataset):
    def __init__(self, fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()


        all_data = []
        for i in range(0, len(lines), 3):
            text = lines[i].strip()
            target = lines[i + 1].strip()
            stance = lines[i + 2].strip()
            seq_len = len(text.split())

            text_indices = tokenizer.text_to_sequence(text)
            target_indices = tokenizer.text_to_sequence(target)
            stance = int(stance) + 1

            data = {
                'text': text,
                'text_len': seq_len,
                'target': target,
                'text_indices': text_indices,
                'target_indices': target_indices,
                'stance': stance,
            }

            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class RoBERTaDataset(Dataset):
    def __init__(self, fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()


        all_data = []
        for i in range(0, len(lines), 3):
            text = lines[i].strip()
            target = lines[i + 1].strip()
            stance = lines[i + 2].strip()

            stance = int(stance) + 1

            data = {
                'text': text,
                'target': target,
                'stance': stance,
            }

            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def statisCount(self, sentence):
        # statis = []
        # statis.append(sentence.count("#"))
        # statis.append(sentence.count("?"))
        # statis.append(len(sentence))
        strCount = str(len(sentence)) + " " + str(sentence.count("#")) + " " + str(sentence.count("?"))
        return strCount

class DatesetReader:
    @staticmethod
    def __read_text__(fnames):
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_raw = lines[i].lower().strip()
                text += text_raw + " "
        return text

    @staticmethod
    def __read_data__(fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = []
        for i in range(0, len(lines), 3):
            text = lines[i].lower().strip()
            target = lines[i + 1].lower().strip()
            stance = lines[i + 2].strip()
            seq_len = len(text.split())
            if seq_len == 0:
                continue


            text_indices = tokenizer.text_to_sequence(text)
            target_indices = tokenizer.text_to_sequence(target)
            stance = int(stance) + 1


            data = {
                'text': text,
                'text_len': seq_len,
                'target': target,
                'text_indices': text_indices,
                'target_indices': target_indices,
                'stance': stance,
            }

            all_data.append(data)
        return all_data

    def __init__(self, dataset='weibo', embed_dim=300, w2v_name='glove-chinese-weibo'):
        print("preparing {0} dataset ...".format(dataset))
        fname = {
            'weibo': {
                'train': './datasets/weibo-train.raw',
                'test': './datasets/weibo-test.raw'
            },
            'SC': {
                'train': './datasets/SC-train.raw',
                'test': './datasets/SC-test.raw'
            },
            # 'twitter': {
            #     'train': './datasets/acl-14-short-data/train.raw',
            #     'test': './datasets/acl-14-short-data/test.raw'
            # },
        }
        text = DatesetReader.__read_text__([fname[dataset]['train'], fname[dataset]['test']])
        if os.path.exists(dataset + '_word2idx.pkl'):
            print("loading {0} tokenizer...".format(dataset))
            with open(dataset + '_word2idx.pkl', 'rb') as f:
                word2idx = pickle.load(f)
                tokenizer = Tokenizer(word2idx=word2idx)
        else:
            tokenizer = Tokenizer()
            tokenizer.fit_on_text(text)
            with open(dataset + '_word2idx.pkl', 'wb') as f:
                pickle.dump(tokenizer.word2idx, f)
        self.embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim, dataset, w2v_name)
        self.train_data = Dataset(DatesetReader.__read_data__(fname[dataset]['train'], tokenizer))
        self.test_data = Dataset(DatesetReader.__read_data__(fname[dataset]['test'], tokenizer))


class Dataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


