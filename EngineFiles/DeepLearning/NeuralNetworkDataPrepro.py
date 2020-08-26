import numpy
import pandas as pd
import random
from sklearn.model_selection import train_test_split

def splitDataset(dataframe=pd.DataFrame(), test_size=0.2, rand_state=42):
    df = dataframe.copy()
    df.dropna(inplace=True)
    df['label'] = df['label'].apply(lambda x : 1 if x == 'positive' else 0)

    _, _, index_train, index_test = train_test_split(df['Tweet'].index, df['label'].index, test_size=0.2, random_state=42, stratify= df.label.values)

    df_train = df.loc[index_train]
    df_test = df.loc[index_test]

    x_train_pos = list(df_train[df_train['label']==1]['Tweet'].values)
    x_test_pos = list(df_test[df_test['label']==1]['Tweet'].values)

    x_train_neg = list(df_train[df_train['label']==0]['Tweet'].values)
    x_test_neg = list(df_test[df_test['label']==0]['Tweet'].values)

    x_train = list(df['Tweet'].loc[index_train].values)
    x_test = list(df['Tweet'].loc[index_test].values)

    y_train = df['label'].loc[index_train].values
    y_test = df['label'].loc[index_test].values

    return df_train, df_test, x_train, x_train_pos, x_train_neg, x_test, x_test_pos, x_test_neg, y_train, y_test, index_train, index_test

def createVocab(x, wiki):
    vocab = {'__PAD__':0, '__</e>__':1, '__UNK__':2}

    for i in [x, wiki]:
        for t in i:
            for w in t.split():
                if w not in vocab:
                    vocab[w] = len(vocab)

    return vocab

def tweet2tensor(tweet, vocabulary, unk_token='__UNK__'):
    word = tweet.split()
    tensor = []
    unk = vocabulary[unk_token]

    for w in word:
        w_id = vocabulary[w] if w in vocabulary else unk
        tensor.append(w_id)

    return tensor

def batch_generator(data_pos, data_neg, batch_size, loop, vocabulary, shuffle=False):
    assert batch_size % 2 == 0
    n = batch_size // 2
    p_index = 0
    n_index = 0
    len_p = len(data_pos)
    len_n = len(data_neg)
    p_index_lines = list(range(len_p))
    n_index_lines = list(range(len_n))

    if shuffle:
        random.shuffle(p_index_lines)
        random.shuffle(n_index_lines)

    stop = False

    while not stop:
        batch = []
        for i in range(n):
            if p_index >= len_p:
                if not loop:
                    stop = True
                    break
                p_index = 0
                if shuffle:
                    random.shuffle(p_index_lines)
            text = data_pos[p_index_lines[p_index]]
            tensor = tweet2tensor(tweet=text, vocabulary=vocabulary)
            batch.append(tensor)
            p_index += 1
        for i in range(n):
            if n_index >= len_n:
                if not loop:
                    stop = True
                    break
                n_index = 0
                if shuffle:
                    random.shuffle(n_index_lines)
            text = data_neg[n_index_lines[n_index]]
            tensor = tweet2tensor(tweet=text, vocabulary=vocabulary)
            batch.append(tensor)
            n_index += 1
        if stop:
            break
        p_index += n
        n_index += n
        max_len = max([len(b) for b in batch])
        tensor_pad = []
        for tns in batch:
            n_pad = max_len - len(tns)
            pad = [0] * n_pad
            t_pad = tns + pad
            tensor_pad.append(t_pad)
        inputs = numpy.array(tensor_pad)
        target_p = [1] * n
        target_n = [0] * n
        target = target_p + target_n
        target = numpy.array(target)
        exm_weights = numpy.ones_like(target)

        yield inputs, target, exm_weights

def train_generator(x, y, vocab, batch_size, shuffle=False):
    return batch_generator(data_pos=x, data_neg=y, batch_size=batch_size, loop=True, vocabulary=vocab, shuffle=shuffle)

def val_generator(x, y, vocab, batch_size, shuffle=False):
    return batch_generator(data_pos=x, data_neg=y, batch_size=batch_size, loop=True, vocabulary=vocab, shuffle=shuffle)

def test_generator(x, y, vocab, batch_size, shuffle=False):
    return batch_generator(data_pos=x, data_neg=y, batch_size=batch_size, loop=False, vocabulary=vocab, shuffle=shuffle)