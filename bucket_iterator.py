# -*- coding: utf-8 -*-

import math
import random
import torch
import numpy


class BucketIterator(object):
    def __init__(self, data, batch_size, sort_key='text_indices', shuffle=True, sort=True):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i * batch_size: (i + 1) * batch_size]))
        return batches

    def pad_data(self, batch_data):
        batch_text = []
        batch_text_len = []
        batch_target = []
        batch_text_indices = []
        batch_target_indices = []
        batch_stance = []

        max_len1 = max([len(t[self.sort_key]) for t in batch_data])
        max_len2 = max([len(t['target_indices']) for t in batch_data])
        max_len = max(max_len1, max_len2)
        for item in batch_data:
            text, seq_len, target, text_indices, target_indices, stance = \
                item['text'], item['text_len'], item['target'], item['text_indices'], item['target_indices'], item['stance']
            text_padding = [0] * (max_len - len(text_indices))
            target_padding = [0] * (max_len - len(target_indices))
            batch_text.append(text)
            batch_text_len.append(seq_len)
            batch_target.append(target)
            batch_text_indices.append(text_indices + text_padding)
            batch_target_indices.append(target_indices + target_padding)
            batch_stance.append(stance)
            # print('batch_text:{}'.format(batch_text))
            #
            # print('batch_target:{}'.format(batch_target))
            #
            # print('batch_target_indices:{}'.format(batch_target_indices))

        return {
            'text': batch_text,
            'text_len': torch.tensor(batch_text_len),
            'target': batch_target,
            'text_indices': torch.tensor(batch_text_indices),
            'target_indices': torch.tensor(batch_target_indices),
            'stance': torch.tensor(batch_stance),
        }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]




class BertBucketIterator(BucketIterator):
    def __init__(self, *args, **kwargs):
        super(BertBucketIterator, self).__init__(*args, **kwargs)

    def pad_data(self, batch_data):
        batch_text = []
        batch_target = []
        batch_text_indices = []
        batch_target_indices = []
        batch_stance = []
        batch_in_graph = []
        batch_cross_graph = []
        batch_bert_segments = []
        batch_len = []

        max_len = max([len(t[self.sort_key]) for t in batch_data])
        for item in batch_data:
            text_indices, attention_mask, bert_segments_ids, in_graph, text_len, stance = \
                item['text_bert_indices'], item['attention_mask'], item['bert_segments_ids'], \
                item['adj_matrix'], item['text_len'], item['stance']
            text_padding = [0] * (max_len - len(text_indices))
            # target_padding = [0] * (max_len - len(target_indices))
            attention_padding = [0] * (max_len - len(text_indices))
            bert_segments_padding = [0] * (max_len - len(text_indices))

            batch_text_indices.append(text_indices)
            batch_target_indices.append(attention_mask)
            batch_bert_segments.append(bert_segments_ids)
            batch_len.append(text_len)
            batch_in_graph.append(numpy.pad(in_graph,
                                            ((0, max_len - len(text_indices)), (0, max_len - len(text_indices))),
                                            'constant'))
            batch_stance.append(stance)
            # print("max_len:{}".format(max_len))
            #
            # print("text_indices.shape:{}".format(text_indices.shape))
            # print("attention_mask.shape:{}".format(attention_mask.shape))
            #
            # print("in_graph.shape:{}".format(in_graph.shape))
            # print("padding后:{}".format(numpy.pad(in_graph,
            #     ((0,max_len-len(text_indices)),(0,max_len-len(text_indices))), 'constant').shape))

            # print("in_graph:{}".format(in_graph.shape))
            # print("numpy.pad(in_graph):{}".format(numpy.pad(in_graph,
            #                                                 ((0, max_len - len(text_indices)),
            #                                                  (0, max_len - len(text_indices))), 'constant').shape))

        # for i in range(len(batch_in_graph)):
        #     in_graph = batch_in_graph[i]
        #     cross_graph = batch_cross_graph[i]
        #     print('dependency_graph:', batch_in_graph[i].shape)
        #     print('sentic_graph:', batch_in_graph[i].shape)

        return { \
            'text_bert_indices': torch.tensor(batch_text_indices), \
            'attention_mask': torch.tensor(batch_target_indices), \
            'bert_segments_ids': torch.tensor(batch_bert_segments), \
            'in_graph': torch.tensor(batch_in_graph), \
            'text_len': torch.tensor(batch_len), \
            'stance': torch.tensor(batch_stance),
            }



class BRoBERTBucketIterator(BucketIterator):
    def __init__(self, *args, **kwargs):
        super(BRoBERTBucketIterator, self).__init__(*args, **kwargs)

    def pad_data(self, batch_data):
        batch_text = []
        batch_target = []
        batch_stance = []
        for item in batch_data:
            text, target, stance = item['text'], item['target'], item['stance']


            batch_text.append(text)
            batch_target.append(target)
            batch_stance.append(stance)



        return { \
            'text': batch_text, \
            'target': batch_target, \
            'stance': batch_stance, \
            }

