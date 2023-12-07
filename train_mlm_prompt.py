#! -*- coding:utf-8 -*-
from pathlib import Path
import os, json
import numpy as np
from bert4keras.backend import keras, set_gelu, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Lambda, Dense
from sklearn import metrics
import jieba
import random
import tensorflow as tf
from bert4keras.layers import Loss

jieba.initialize()
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

modelPath = os.path.dirname(os.path.realpath(__file__))
lr = 1e-5
num_classes = 3
maxlen = 128
batch_size = 16

# bert配置
config_path = modelPath + '/word-embedding/pretrain/wobert-mlm-weibo/bert_config.json'
checkpoint_path = modelPath + '/word-embedding/pretrain/wobert-mlm-weibo/bert_model.ckpt'
# dict_path = modelPath + '/word-embedding/pretrain/wobert-zh/vocab.txt'


def seed_tensorflow(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_random_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


seed_tensorflow()


def load_data(filename):
    D = []
    with open(filename, 'r', encoding='utf8') as train:
        for line in train:
            if len(line.split("\t")) != 4: continue
            line = line.replace('\n', '').replace('\r', '')
            if line.split("\t")[2] == 'TEXT': continue

            if line.split("\t")[3] == "FAVOR":
                stance = 2
            elif line.split("\t")[3] == "AGAINST":
                stance = 0
            else:
                stance = 1
            D.append((line.split("\t")[2][:110], stance, line.split("\t")[1]))
    return D


# 加载数据集
train_data = load_data(modelPath + '/datasets/weibo-train')
valid_data = load_data(modelPath + '/datasets/weibo-test')
test_data = load_data(modelPath + '/datasets/weibo-test')

# Few-shot
random.shuffle(train_data)
# train_data = train_data[:32]
print('train_data_size={}'.format(len(train_data)))
# 建立分词器
token_dict, keep_tokens, compound_tokens = json.load(
    open('tokenizer_config.json')
)

tokenizer = Tokenizer(
    token_dict,
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.cut(s, HMM=False)
)

# 对应的任务描述
prompt = u'所以 我 对于 - 的 态度 是 - .'

against_id = tokenizer.token_to_id(u'反对')  # 36326
none_id = tokenizer.token_to_id(u'中立')  # 24550
favour_id = tokenizer.token_to_id(u'支持')  # 36257


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):

        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, (text, label, target) in self.sample(random):
            prompt_target = prompt.replace('-', target, 1)
            # print(prompt_target)
            token_ids, segment_ids = tokenizer.encode(text, prompt_target, maxlen=maxlen)

            source_ids, target_ids = token_ids[:], token_ids[:]
            mask_idx = len(source_ids) - 3  # ⭐️ 定位[mask]的位置 [CLS]: 101 [SEP]: 102, 得基于prompt来修改
            # mask_idx = source_ids.index(102) + 3  # ⭐️ 定位[mask]的位置 [CLS]: 101 [SEP]: 102, 得基于prompt来修改
            if label == 0:
                source_ids[mask_idx] = tokenizer._token_mask_id
                target_ids[mask_idx] = against_id
            elif label == 1:
                source_ids[mask_idx] = tokenizer._token_mask_id
                target_ids[mask_idx] = none_id
            elif label == 2:
                source_ids[mask_idx] = tokenizer._token_mask_id
                target_ids[mask_idx] = favour_id
            # print(tokenizer.ids_to_tokens(source_ids))
            # print(tokenizer.ids_to_tokens(target_ids))
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(target_ids)
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
        print(y_true)
        print('=' * 50)
        print(y_pred)
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        accuracy = K.sum(accuracy * y_mask) / K.sum(y_mask)
        self.add_metric(accuracy, name='accuracy')
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


# 加载预训练模型
model = build_transformer_model(
    config_path,
    checkpoint_path,
    # keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    # compound_tokens=compound_tokens,  # 增加词，用字平均来初始化
    with_mlm=True,
)
# 训练用模型
y_in = keras.layers.Input(shape=(None,))
outputs = CrossEntropy(1)([y_in, model.output])


print(f'lr={lr}')
train_model = keras.models.Model(model.inputs + [y_in], outputs)
train_model.compile(
    optimizer=Adam(lr),  # 用足够小的学习率
)
train_model.summary()

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
# test_generator = data_generator(test_data, batch_size)


# 通过倒叙遍历列表来定位[mask]的位置
def find_mask_position(list):
    for i in range(len(list) - 1, -1, -1):
        if int(list[i]) != 0:
            return i - 2


def get_AOC(y_label, y_score):
    from sklearn.metrics import roc_curve, auc

    # y_label = np.array([
    #     [1, 0, 0], [1, 0, 0], [1, 0, 0],
    #     [0, 1, 0], [0, 1, 0], [0, 1, 0],
    #     [0, 0, 1], [0, 0, 1], [0, 0, 1]
    # ])
    #
    # y_score = np.array([
    #     [0.8, 0.1, 0.1], [0.2, 0.32, 0.48], [0.6, 0.1, 0.3],
    #     [0.2, 0.5, 0.3], [0.1, 0.6, 0.3], [0.2, 0.75, 0.05],
    #     [0.05, 0.05, 0.9], [0.1, 0.3, 0.6], [0.12, 0.8, 0.08],
    # ])

    n_classes = 3

    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_label[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_label.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # print('fpr["micro"]={}'.format(fpr["micro"].tolist()))
    # print('tpr["micro"]={}'.format(tpr["micro"].tolist()))
    # print('microAUC={}'.format(roc_auc["micro"]))

    # macro（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # print('fpr["macro"]={}'.format(fpr["macro"].tolist()))
    # print('tpr["macro"]={}'.format(tpr["macro"].tolist()))
    print('macraAUC={}'.format(roc_auc["macro"]))


def get_AP(Y_test, y_score):
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score

    # (1) For each class
    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(3):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                            y_score[:, i])

        average_precision[i] = average_precision_score(Y_test[:, i],
                                                       y_score[:, i])

    # (2) A "macro-average": quantifying score on all classes jointly
    precision["macro"], recall["macro"], _ = precision_recall_curve(Y_test.ravel(),
                                                                    y_score.ravel())
    # print('precision["macro"]={}'.format(precision["macro"].tolist()))
    # print('recall["macro"]={}'.format(recall["macro"].tolist()))
    average_precision["macro"] = average_precision_score(Y_test, y_score,
                                                         average="macro")

    print('Average precision score, macro-averaged over all classes: {0:0.4f}'.format(average_precision["macro"]))


def evaluate(data, is_test=False):
    y_trues, y_preds = [], []
    y_scores = []
    y_pred_items = []
    for x_true, _ in data:
        x_true, y_true = x_true[:2], x_true[2]
        y_pred_item = model.predict(x_true)

        for i in range(len(y_true)):       # ⭐️ 记得修改和上面对应

            pred_all = y_pred_item[i, find_mask_position(y_true[i].tolist()), [against_id, none_id, favour_id]]  # 三类别的概率

            y_scores.append(pred_all)
            pred = pred_all.argmax(axis=0)  # 选概率大的那一个词
            y_preds.append(pred)

            true = y_true[i, find_mask_position(y_true[i].tolist())]  # pos的标签字为1，[0 0 1 1 0 1 0 01 0 0]

            if true == against_id:
                y_trues.append(0)
            elif true == none_id:
                y_trues.append(1)
            elif true == favour_id:
                y_trues.append(2)
        # print(y_preds)
        # print(y_trues)
    print('train_data_size={}'.format(len(train_data)))
    f1_mac = metrics.f1_score(y_trues, y_preds, labels=[0, 2], average='macro')
    f1_mic = metrics.f1_score(y_trues, y_preds, labels=[0, 2], average='micro')

    acc_sk = metrics.accuracy_score(y_trues, y_preds)
    precision = metrics.precision_score(y_trues, y_preds, labels=[0, 2], average='macro')
    recall = metrics.recall_score(y_trues, y_preds, labels=[0, 2], average='macro')
    class_result = metrics.classification_report(y_trues, y_preds, target_names=['反对', '中立', '支持'], digits=4)
    print(class_result)
    print(
        '>> test_acc: {:.4f}, test_precision:{:.4f}, test_recall:{:.4f}, test_f1_mac: {:.4f}, test_f1_mic: {:.4f}'.format(acc_sk, precision,
                                                                                                 recall, f1_mac, f1_mic))

    # if is_test:
    #     eval_each_target_Z_stance(y_trues, y_preds)

    i = 0
    for item in y_trues:
        if item == 0:
            y_trues[i] = [1, 0, 0]
        elif item == 1:
            y_trues[i] = [0, 1, 0]
        elif item == 2:
            y_trues[i] = [0, 0, 1]
        i = i + 1
    # 计算交叉熵损失
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    loss = loss_fn(np.array(y_trues), np.array(y_scores))
    # print(loss.numpy())
    # print(K.eval(K.cast(loss, dtype='float32')))
    print('test_loss: {:.4f}'.format(K.eval(K.cast(loss, dtype='float32'))))
    # get_AOC(np.array(y_trues), np.array(y_scores))
    # get_AP(np.array(y_trues), np.array(y_scores))
    return f1_mac

def eval_each_target(y_trues, y_preds):
    target_list = ['CJ_T2', 'SE_T1', 'FK_T3', 'ET_T4', 'SZ_T5']
    count = 0
    for i in range(0, len(y_trues), 200):
        f1_mac = metrics.f1_score(y_trues[i:i+200], y_preds[i:i+200], labels=[0, 2], average='macro')
        print('{}:'.format(target_list[count]))
        print(u'val_f1: %.4f\n' % (f1_mac))
        count = count + 1

def eval_each_target_Z_stance(y_trues, y_preds):
    import pandas as pd
    # target_list = ['CJ_T2', 'SE_T1', 'FK_T3', 'ET_T4', 'SZ_T5']
    target_name_list = ['三胎生育政策来了', '俄乌冲突', '华为鸿蒙', '滴滴下架']
    target_list = []
    i = 0
    with open(modelPath+'/datasets/Z-stance-dev.txt', 'r', encoding='utf8') as train:
        for line in train:
            if len(line.split("\t")) != 4: continue
            line = line.replace('\n', '').replace('\r', '')
            if line == 'TARGET':
                continue

            target_list.append(line.split("\t")[1])
    # print(len(target_list))
    # print(target_list)
    # print(len(y_trues))
    # print(y_trues)
    data = {'target': target_list[1:],
            'y_trues': y_trues,
            'y_preds': y_preds
            }
    df = pd.DataFrame(data)
    for item in target_name_list:
        df_target = df[df['target'] == item]
        f1_mac = metrics.f1_score(df_target['y_trues'], df_target['y_preds'], labels=[0, 2], average='macro')
        print('{}:'.format(item))
        print(u'val_f1: %.4f\n' % (f1_mac))
        class_result = metrics.classification_report(df_target['y_trues'], df_target['y_preds'], target_names=['反对', '中立', '支持'], digits=4)
        print(class_result)




class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_f1 = evaluate(valid_generator, True)
        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
            model.save_weights(modelPath + '/output/keras-prompt/best_model_stance_fine_tune.weights')
        # test_f1 = evaluate(test_generator)
        print(u'val_f1: %.4f, best_val_f1: %.4f\n' % (val_f1, self.best_val_f1))


if __name__ == '__main__':

    evaluator = Evaluator()

    train_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=50,
        callbacks=[evaluator]
    )

    model.load_weights(modelPath + '/output/keras-prompt/best_model_stance_fine_tune.weights')
    print(u'final test acc: %04f\n' % (evaluate(valid_generator, True)))

else:

    model.load_weights(modelPath + '/output/keras-prompt/best_model_stance_fine_tune.weights')
