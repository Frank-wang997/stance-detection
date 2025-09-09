#! -*- coding: utf-8 -*-
# 词级别的中文BERT预训练
# MLM任务

import argparse
import os, json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.optimizers import extend_with_weight_decay
from bert4keras.optimizers import extend_with_gradient_accumulation
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator
from bert4keras.snippets import text_segmentate
import jieba
jieba.initialize()

modelPath = os.path.dirname(os.path.realpath(__file__))

# 基本参数
maxlen = 512
batch_size = 16
epochs =  100
num_words = 30000
parser = argparse.ArgumentParser()
parser.add_argument('--topic_augment', default=False, type=bool)  # 是否进行主题增强训练
opt = parser.parse_args()

# bert配置
if opt.topic_augment:
    config_path = modelPath+'/word-embedding/pretrain/wobert-mlm-weibo/bert_config.json'
    checkpoint_path = modelPath+'/word-embedding/pretrain/wobert-mlm-weibo/bert_model.ckpt'
    dict_path = modelPath+'/word-embedding/pretrain/wobert-zh/vocab.txt'
else:
    config_path = modelPath + '/word-embedding/pretrain/wobert-zh/bert_config.json'
    checkpoint_path = modelPath + '/word-embedding/pretrain/wobert-zh/bert_model.ckpt'
    dict_path = modelPath + '/word-embedding/pretrain/wobert-zh/vocab.txt'


def corpus():
    """语料生成器
    """
    while True:
        f = modelPath+'/datasets/data_ori/weibo_target.json'
        with open(f) as f:
            for l in f:
                l = json.loads(l)
                for text in text_process(l['text']):
                    yield text, l['target']


def text_process(text):
    """分割文本
    """
    texts = text_segmentate(text, 32, u'\n。')
    result, length = '', 0
    for text in texts:
        if result and len(result) + len(text) > maxlen * 1.3:
            yield result
            result, length = '', 0
        result += text
    if result:
        yield result


# 获取每个目标的seed_word
def load_seed_word(path, weightFilter=0.5):
    seed_words = {}
    words = []
    fp = open(path, 'r')
    for line in fp:
        line = line.strip()
        if not line:
            continue
        word, weight = line.split('\t')
        if float(weight) < weightFilter:
            continue
        words.append(word)
        seed_words[word] = weight
    fp.close()
    return words

all_words_seed = []
all_words_seed.append(load_seed_word(modelPath+'/seed_words/seed_word.SE.label.stance'))
all_words_seed.append(load_seed_word(modelPath+'/seed_words/seed_word.CJ.label.stance'))
all_words_seed.append(load_seed_word(modelPath+'/seed_words/seed_word.FK.label.stance'))
all_words_seed.append(load_seed_word(modelPath+'/seed_words/seed_word.ET.label.stance'))
all_words_seed.append(load_seed_word(modelPath+'/seed_words/seed_word.SZ.label.stance'))


TARGET_DIC = {'iphonese': all_words_seed[0], '春节放鞭炮': all_words_seed[1], '俄罗斯叙利亚反恐行动': all_words_seed[2], '开放二胎': all_words_seed[3], '深圳禁摩限电': all_words_seed[4]}

import operator
from functools import reduce
# 所有词
all_word = reduce(operator.add, all_words_seed)

if os.path.exists('tokenizer_config.json'):
    token_dict, keep_tokens, compound_tokens = json.load(
        open('tokenizer_config.json')
    )
else:
    # 加载并精简词表
    token_dict, keep_tokens = load_vocab(
        dict_path=dict_path,
        simplified=True,
        startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
    )
    pure_tokenizer = Tokenizer(token_dict.copy(), do_lower_case=True)
    user_dict = []
    for w in all_word:
        if w not in token_dict:
            print("新词：{}".format(w))
            token_dict[w] = len(token_dict)
            user_dict.append(w)
        if len(user_dict) == num_words:
            break
    if len(user_dict) < num_words:
        for w, _ in sorted(jieba.dt.FREQ.items(), key=lambda s: -s[1]):
            if w not in token_dict:
                print("新词：{}".format(w))
                token_dict[w] = len(token_dict)
                user_dict.append(w)
            if len(user_dict) == num_words:
                break
    compound_tokens = [pure_tokenizer.encode(w)[0][1:-1] for w in user_dict]
    json.dump([token_dict, keep_tokens, compound_tokens],
              open('tokenizer_config.json', 'w'))

tokenizer = Tokenizer(
    token_dict,
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.cut(s)
)


# print(all_words_seed[0])
# se = tokenizer.tokens_to_ids(all_words_seed[0])
# print(se)

def random_masking(token_ids):
    """对输入进行随机mask
    """
    rands = np.random.random(len(token_ids))
    source, target = [], []
    for r, t in zip(rands, token_ids):
        if r < 0.15 * 0.8:
            source.append(tokenizer._token_mask_id)
            target.append(t)
        elif r < 0.15 * 0.9:
            source.append(t)
            target.append(t)
        elif r < 0.15:
            source.append(np.random.choice(tokenizer._vocab_size - 1) + 1)
            target.append(t)
        else:
            source.append(t)
            target.append(0)
    # print(tokenizer.ids_to_tokens(source))
    # print(tokenizer.ids_to_tokens(target))
    return source, target


def target_masking(token_ids, text_target, mask_ratio=0.4):
    """对输入进行关键词mask
    """
    special_ids = {tokenizer._token_cls_id, tokenizer._token_sep_id}
    candidates = [i for i, t in enumerate(token_ids) if t not in special_ids]

    target_mask_count = max(1, int(round(len(candidates) * mask_ratio)))

    seeds = TARGET_DIC.get(text_target, [])
    seed_ids = set(tokenizer.tokens_to_ids(seeds))
    seed_positions = [i for i in candidates if token_ids[i] in seed_ids]
    np.random.shuffle(seed_positions)

    chosen = list(seed_positions[:target_mask_count])

    if len(chosen) < target_mask_count:
        remain_pool = [i for i in candidates if i not in chosen]
        np.random.shuffle(remain_pool)
        need = target_mask_count - len(chosen)
        chosen += remain_pool[:need]

    mask_positions = set(chosen)

    source, target = [], []
    for i, t in enumerate(token_ids):
        if i in mask_positions:
            source.append(tokenizer._token_mask_id)
            target.append(t)
        else:
            source.append(t)
            target.append(0)

    return source, target



class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, text in self.sample(random):
            # print(text)
            token_ids, segment_ids = tokenizer.encode(text[0], maxlen=maxlen)
            # bert配置
            if opt.topic_augment:
                source, target = target_masking(token_ids, text[1])
            else:
                source, target = random_masking(token_ids)

            batch_token_ids.append(source)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(target)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_output_ids = sequence_padding(batch_output_ids)
                yield [
                    batch_token_ids, batch_segment_ids, batch_output_ids
                ], None
                batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        accuracy = K.sum(accuracy * y_mask) / K.sum(y_mask)
        self.add_metric(accuracy, name='accuracy')
        loss = K.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss

# print(len(keep_tokens))
bert = build_transformer_model(
    config_path,
    checkpoint_path,
    with_mlm='linear',
    # keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    compound_tokens=compound_tokens,  # 增加词，用字平均来初始化
    return_keras_model=False
)

# 训练用模型
y_in = keras.layers.Input(shape=(None,))
outputs = CrossEntropy(1)([y_in, bert.model.output])

train_model = keras.models.Model(bert.model.inputs + [y_in], outputs)

AdamW = extend_with_weight_decay(Adam, name='AdamW')
AdamWG = extend_with_gradient_accumulation(AdamW, name='AdamWG')
optimizer = AdamWG(
    learning_rate=5e-6,
    weight_decay_rate=0.01,
    exclude_from_weight_decay=['Norm', 'bias'],
    grad_accum_steps=16,
)
train_model.compile(optimizer=optimizer)
train_model.summary()


class Evaluator(keras.callbacks.Callback):
    """训练回调
    """
    def on_epoch_end(self, epoch, logs=None):
        # model.save_weights('bert_model.weights')  # 保存模型
        if opt.topic_augment:
            bert.save_weights_as_checkpoint(modelPath+'/word-embedding/pretrain/wobert-mlm-weibo/bert_model_target.ckpt')  # 保存模型
        else:
            bert.save_weights_as_checkpoint(
                modelPath + '/word-embedding/pretrain/wobert-mlm-weibo/bert_model.ckpt')  # 保存模型


if __name__ == '__main__':




    # 启动训练
    evaluator = Evaluator()
    train_generator = data_generator(corpus(), batch_size, 10**5)

    train_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=1000,
        epochs=epochs,
        callbacks=[evaluator]
    )

else:

    if opt.topic_augment:
        bert.save_weights_as_checkpoint(
            modelPath + '/word-embedding/pretrain/wobert-mlm-weibo/bert_model_target.ckpt')  # 保存模型
    else:
        bert.save_weights_as_checkpoint(
            modelPath + '/word-embedding/pretrain/wobert-mlm-weibo/bert_model.ckpt')  # 保存模型