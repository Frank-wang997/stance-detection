#!/usr/bin/env python3


import nltk
import math
import numpy as np
import spacy
# nlp = spacy.load('zh_core_web_sm')
import jieba


# TARGET_DIC = {'iphonese': 0, '春节放鞭炮': 1, '俄罗斯叙利亚反恐行动': 2, '开放二胎': 3, '深圳禁摩限电': 4}
TARGET_DIC = {'三胎生育政策来了': 0, '俄乌冲突': 1, '滴滴下架': 2, '华为鸿蒙': 3}

INDEX = 3

LABEL_DIC = {'AGAINST': 0, 'FAVOR': 1}

def process(filename):
    fin = open(filename, 'r', encoding='utf-8', errors='ignore')
    fout = open('Z-stance/seed_word.HW.stance', 'w', encoding='utf-8')
    word_freq_dic = {}
    word_stance_dic = {}
    label_count = [0, 0]
    for line in fin:
        line = line.strip()
        if not line:
            continue
        item = line.split('\t')
        target = item[1]

        # print("target:{}".format(target))
        if target not in TARGET_DIC:
            print('NOT IN TARGET_DIC:{}'.format(target))
            continue
        stance = item[-1]
        # print("stance:{}".format(stance))
        #if stance not in LABEL_DIC:
            #continue
        if stance in LABEL_DIC:
            label_count[LABEL_DIC[stance]] += 1
        text = ' '.join(item[2:-1])

        text = jieba.cut(text, HMM=False)
        # doc = nlp(text)
        # text = ' '.join([t.text for t in doc])
        text = list(text)
        text_cleaned = []
        for item in text:
            if '\u4e00' <= item <= '\u9fa5':
                text_cleaned.append(item)
            elif item.isalpha():
                text_cleaned.append(item)


        # print("text:{}".format(list(text_cleaned)))
        for word in set(text_cleaned):
            if word not in word_freq_dic:
                word_freq_dic[word] = [0, 0, 0, 0, 0]
                word_freq_dic[word][TARGET_DIC[target]] += 1
            else:
                word_freq_dic[word][TARGET_DIC[target]] += 1
            # print(word_freq_dic)
            if stance not in LABEL_DIC:
                continue
            if word not in word_stance_dic:
                word_stance_dic[word] = [0, 0]
                word_stance_dic[word][LABEL_DIC[stance]] += 1
            else:
                word_stance_dic[word][LABEL_DIC[stance]] += 1
    fin.close()

    stance_weight_list = []
    stance_weight_dic = {}
    max_stance_weight = 0
    min_stance_weight = 100000
    for word in word_stance_dic:
        freq_list = word_stance_dic[word]
        weight = freq_list[0] - freq_list[1]
        weight = freq_list[0] / label_count[0] - freq_list[1] / label_count[1]
        stance_weight_dic[word] = weight
        stance_weight_list.append(weight)
        if weight > max_stance_weight:
            max_stance_weight = weight
        if weight < min_stance_weight:
            min_stance_weight = weight

    print(max_stance_weight, min_stance_weight)
    mu = np.mean(stance_weight_list)
    sigma = np.std(stance_weight_list)
    _range = np.max((np.abs(stance_weight_list)))

    min_weight = 100000
    max_weight = 0
    seed_weight = {}
    for word in word_freq_dic:
        freq_list = word_freq_dic[word]
        weight = freq_list[INDEX] / (sum(freq_list)+1)
        seed_weight[word] = weight
        if weight > max_weight:
            max_weight = weight
        if weight < min_weight:
            min_weight = weight
    seed_weight = sorted(seed_weight.items(), key= lambda a: -a[1])
    
    for (seed, weight) in seed_weight:
        if seed in stance_weight_dic:
            stance_weight = (stance_weight_dic[seed]-mu) / sigma
            stance_weight = 1 + stance_weight_dic[seed] / _range
            #stance_weight = -1 + ((2/(max_stance_weight-min_stance_weight)) * (stance_weight_dic[seed]-min_stance_weight))
            # print(seed, stance_weight)
        else:
            stance_weight = 1
        weight = (weight-min_weight) / (max_weight-min_weight)
        # 下一行的代码没有在本论文中应用（因在少样本场景）
        # weight *= stance_weight
        string = seed + '\t' + str(weight) + '\n'
        fout.write(string)
    fout.close()




if __name__ == '__main__':

    process('../datasets/Z-stance.txt')
