#! -*- coding:utf-8 -*-
import os, json, random
import numpy as np
import jieba
import tensorflow as tf
from pathlib import Path
from sklearn import metrics

from bert4keras.backend import keras, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator, open
from keras.layers import Input, Lambda, Add, Layer
from keras.models import Model

jieba.initialize()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
modelPath = os.path.dirname(os.path.realpath(__file__))

seed = 42
random.seed(seed); np.random.seed(seed);
try: tf.random.set_seed(seed)
except Exception: pass
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

lr = 1e-5
maxlen = 128
batch_size = 16
lambda_stance = 0.8

num_stance = 3
num_sentiment = 3

config_path = modelPath + '/word-embedding/pretrain/wobert-mlm-weibo/bert_config.json'
checkpoint_path = modelPath + '/word-embedding/pretrain/wobert-mlm-weibo/bert_model.ckpt'

def _parse_line(line):
    parts = line.rstrip('\n').rstrip('\r').split('\t')
    if len(parts) >= 3 and parts[2] == 'TEXT':
        return None
    if len(parts) >= 5:
        target, text, stance_str, sentiment_str = parts[1], parts[2][:110], parts[3], parts[4]
    elif len(parts) == 4:
        target, text, stance_str, sentiment_str = parts[0], parts[1][:110], parts[2], 'NEUTRAL'
    else:
        return None

    stance_map = {'AGAINST':0, 'NONE':1, 'FAVOR':2}
    sent_map   = {'NEGATIVE':0, 'NEUTRAL':1, 'POSITIVE':2}
    stance = stance_map.get(stance_str.upper(), 1)
    sentiment = sent_map.get(sentiment_str.upper(), 1)
    return (text, target, stance, sentiment)

def load_data(filename):
    D = []
    with open(filename, 'r', encoding='utf8') as f:
        for line in f:
            item = _parse_line(line)
            if item is not None:
                D.append(item)
    return D

def _resolve(paths):
    for p in paths:
        if os.path.exists(p): return p
    return None

train_file = _resolve([
    modelPath + '/datasets/weibo-train',
    modelPath + '/datasets/weibo-train.txt',
    os.path.join(modelPath, 'weibo-train.txt')
])
valid_file = _resolve([
    modelPath + '/datasets/weibo-test',
    modelPath + '/datasets/weibo-test.txt',
    os.path.join(modelPath, 'weibo-test.txt')
])
if train_file is None:
    raise FileNotFoundError("Missing weibo-train(.txt) under ./datasets or current directory.")
if valid_file is None:
    valid_file = train_file

train_data = load_data(train_file)
valid_data = load_data(valid_file)
random.shuffle(train_data)
print(f'train_data_size={len(train_data)}')

token_dict, keep_tokens, compound_tokens = json.load(open('tokenizer_config.json'))
tokenizer = Tokenizer(token_dict, do_lower_case=True, pre_tokenize=lambda s: jieba.cut(s, HMM=False))

vid_against = tokenizer.token_to_id('反对')
vid_none    = tokenizer.token_to_id('中立')
vid_favor   = tokenizer.token_to_id('支持')
if 0 in (vid_against, vid_none, vid_favor):
    raise ValueError('Verbalizers for stance not found in vocab: 反对/中立/支持')

def pick_first_in_vocab(cands):
    for w in cands:
        wid = tokenizer.token_to_id(w)
        if wid != 0:
            return w, wid
    return None, 0

neg_tok, vid_neg   = pick_first_in_vocab(['负面','消极','差','糟'])
neu_tok, vid_neu   = pick_first_in_vocab(['中性','一般','普通','一般般'])
pos_tok, vid_pos   = pick_first_in_vocab(['正面','积极','好','喜欢'])
if 0 in (vid_neg, vid_neu, vid_pos):
    raise ValueError('Could not find suitable Chinese verbalizers for sentiment in vocab.')

dash_id = tokenizer.token_to_id('-')

stance_tpl = '所以 我 对于 - 的 态度 是 - 。'
sent_tpl   = '所以 我 的 心情 是 - 。'

class MTTRPM_MaskDot_Generator(DataGenerator):
    def __iter__(self, random=False):
        s_ids, s_segs, s_pos, s_lab = [], [], [], []
        e_ids, e_segs, e_pos, e_lab = [], [], [], []

        for is_end, (text, target, stance, sentiment) in self.sample(random):
            stance_prompt = stance_tpl.replace('-', target, 1)
            t_ids, t_segs = tokenizer.encode(text, stance_prompt, maxlen=maxlen)

            m_idx = None
            for j in range(len(t_ids)):
                if t_ids[j] == dash_id:
                    m_idx = j; break
            if m_idx is None:
                for j in range(len(t_ids)-1, -1, -1):
                    if int(t_ids[j]) != 0:
                        m_idx = max(j-2, 0); break
            src_ids = t_ids[:]
            src_ids[m_idx] = tokenizer._token_mask_id

            sent_prompt = sent_tpl
            u_ids, u_segs = tokenizer.encode(text, sent_prompt, maxlen=maxlen)
            m2_idx = None
            for j in range(len(u_ids)):
                if u_ids[j] == dash_id:
                    m2_idx = j; break
            if m2_idx is None:
                for j in range(len(u_ids)-1, -1, -1):
                    if int(u_ids[j]) != 0:
                        m2_idx = max(j-2, 0); break
            src_ids2 = u_ids[:]
            src_ids2[m2_idx] = tokenizer._token_mask_id

            s_ids.append(src_ids); s_segs.append(t_segs); s_pos.append([m_idx]); s_lab.append([stance])
            e_ids.append(src_ids2); e_segs.append(u_segs); e_pos.append([m2_idx]); e_lab.append([sentiment])

            if len(s_ids) == self.batch_size or is_end:
                yield [
                    sequence_padding(s_ids),
                    sequence_padding(s_segs),
                    sequence_padding(e_ids),
                    sequence_padding(e_segs),
                    np.array(s_pos, dtype='int32'),
                    np.array(e_pos, dtype='int32'),
                    np.array(s_lab, dtype='int32'),
                    np.array(e_lab, dtype='int32'),
                ], None
                s_ids, s_segs, s_pos, s_lab = [], [], [], []
                e_ids, e_segs, e_pos, e_lab = [], [], [], []

train_gen = MTTRPM_MaskDot_Generator(train_data, batch_size)
valid_gen = MTTRPM_MaskDot_Generator(valid_data, batch_size)

mlm_encoder = build_transformer_model(
    config_path, checkpoint_path,
    with_mlm=True,
)

# inputs
s_token_ids = Input(shape=(None,), dtype='int32', name='stance_token_ids')
s_segment_ids = Input(shape=(None,), dtype='int32', name='stance_segment_ids')
e_token_ids = Input(shape=(None,), dtype='int32', name='sent_token_ids')
e_segment_ids = Input(shape=(None,), dtype='int32', name='sent_segment_ids')
s_pos_in = Input(shape=(1,), dtype='int32', name='stance_mask_pos')
e_pos_in = Input(shape=(1,), dtype='int32', name='sent_mask_pos')
y_s = Input(shape=(1,), dtype='int32', name='y_stance')
y_e = Input(shape=(1,), dtype='int32', name='y_sent')

s_probs = mlm_encoder([s_token_ids, s_segment_ids])
e_probs = mlm_encoder([e_token_ids, e_segment_ids])

vocab_size = K.int_shape(s_probs)[-1]

def gather_mask_prob(args):
    probs, pos = args
    pos = K.reshape(pos, (-1,))
    B = K.shape(probs)[0]
    L = K.shape(probs)[1]
    V = K.shape(probs)[2]
    rng = K.arange(0, B)
    idx = K.stack([rng, pos], axis=1)
    gathered = tf.gather_nd(probs, idx)
    return gathered

s_vec = Lambda(gather_mask_prob, name='gather_stance')([s_probs, s_pos_in])
e_vec = Lambda(gather_mask_prob, name='gather_sent')([e_probs, e_pos_in])

class VocabToClass(Layer):
    def __init__(self, num_classes, name=None):
        super(VocabToClass, self).__init__(name=name)
        self.num_classes = num_classes
    def build(self, input_shape):
        V = int(input_shape[-1])
        self.E = self.add_weight(
            name='E', shape=(V, self.num_classes),
            initializer='glorot_uniform', trainable=True
        )
        super(VocabToClass, self).build(input_shape)
    def call(self, x):
        return K.dot(x, self.E)

stance_logits = VocabToClass(num_stance, name='stance_proj')(s_vec)
sent_logits   = VocabToClass(num_sentiment, name='sent_proj')(e_vec)

def sparse_ce_logits(y_true, y_pred):
    y_true = K.reshape(K.cast(y_true, 'int32'), (-1,))
    return K.mean(K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True))

stance_loss = Lambda(lambda x: sparse_ce_logits(x[0], x[1]), name='stance_loss')([y_s, stance_logits])
sent_loss   = Lambda(lambda x: sparse_ce_logits(x[0], x[1]), name='sent_loss')([y_e, sent_logits])

total_loss = Add(name='total_loss')([
    Lambda(lambda x: x * lambda_stance)(stance_loss),
    Lambda(lambda x: x * (1.0 - lambda_stance))(sent_loss)
])

train_model = Model(
    inputs=[s_token_ids, s_segment_ids, e_token_ids, e_segment_ids, s_pos_in, e_pos_in, y_s, y_e],
    outputs=total_loss
)
train_model.add_metric(keras.metrics.sparse_top_k_categorical_accuracy(K.reshape(y_s, (-1,)), stance_logits, k=1),
                       name='stance_acc', aggregation='mean')
train_model.add_metric(keras.metrics.sparse_top_k_categorical_accuracy(K.reshape(y_e, (-1,)), sent_logits, k=1),
                       name='sent_acc', aggregation='mean')

train_model.compile(optimizer=Adam(lr))
train_model.summary()

infer_model = Model([s_token_ids, s_segment_ids, e_token_ids, e_segment_ids, s_pos_in, e_pos_in],
                    [stance_logits, sent_logits])

def evaluate(gen):
    stance_trues, stance_preds = [], []
    for (s_ids, s_segs, e_ids, e_segs, s_pos, e_pos, y_s_, y_e_), _ in gen:
        s_logit, _ = infer_model.predict([s_ids, s_segs, e_ids, e_segs, s_pos, e_pos])
        pred = s_logit.argmax(axis=-1).tolist()
        y = y_s_.reshape(-1).tolist()
        stance_preds.extend(pred)
        stance_trues.extend(y)

    f1_mac = metrics.f1_score(stance_trues, stance_preds, labels=[0,2], average='macro')
    f1_mic = metrics.f1_score(stance_trues, stance_preds, labels=[0,2], average='micro')
    acc    = metrics.accuracy_score(stance_trues, stance_preds)
    print('STANCE >> acc={:.4f}  f1_mac={:.4f}  f1_mic={:.4f}'.format(acc, f1_mac, f1_mic))
    print(metrics.classification_report(stance_trues, stance_preds, target_names=['反对','中立','支持'], digits=4))
    return f1_mac

class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best = 0.
        self.path = os.path.join(modelPath, 'output/keras-prompt/mttrpm_maskdot_best.weights')
        Path(os.path.dirname(self.path)).mkdir(parents=True, exist_ok=True)
    def on_epoch_end(self, epoch, logs=None):
        f1 = evaluate(valid_gen)
        if f1 > self.best:
            self.best = f1
            mlm_encoder.save_weights(self.path)
        print('val_f1={:.4f}  best={:.4f}\n'.format(f1, self.best))

if __name__ == '__main__':
    evaluator = Evaluator()
    train_model.fit_generator(
        train_gen.forfit(),
        steps_per_epoch=len(train_gen),
        epochs=50,
        callbacks=[evaluator]
    )
    mlm_encoder.load_weights(os.path.join(modelPath, 'output/keras-prompt/mttrpm_maskdot_best.weights'))
    print('final val f1: {:.4f}'.format(evaluate(valid_gen)))
else:
    pass
