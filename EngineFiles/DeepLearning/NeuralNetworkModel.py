from trax import fastmath
from trax import layers as tl
from trax.supervised import training

from sklearn.metrics import accuracy_score, classification_report, f1_score

import EngineFiles.TweetClean as tc
from EngineFiles.DeepLearning import NeuralNetworkDataPrepro as NND

def classifier(vocab_size=1, embedding_dim=256, output_dim=2, mode='train'):
    embed_layer = tl.Embedding(vocab_size=vocab_size, d_feature=embedding_dim)
    mean_layer = tl.Mean(axis=1)
    dense_output_layer = tl.Dense(n_units=output_dim)
    log_softmax_layer = tl.LogSoftmax()
    model = tl.Serial(embed_layer, mean_layer, dense_output_layer, log_softmax_layer)
    return model

def train_model(classifier, train_task, eval_task, n_steps, output_dir):
    training_loop = training.Loop(classifier,
                                    train_task,
                                    eval_tasks=[eval_task],
                                    output_dir=output_dir,
                                    random_seed=31)
    training_loop.run(n_steps=n_steps)
    return training_loop

def compute_accuracy(preds, y, y_weights):
    is_pos = preds[:,1] > preds[:,0]
    is_pos_int = is_pos.astype(fastmath.numpy.int32)
    correct = is_pos_int == y
    sum_weights = fastmath.numpy.sum(y_weights)
    correct_float = correct.astype(fastmath.numpy.float32)
    w_correct_float = correct_float * y_weights
    w_num_correct = fastmath.numpy.sum(w_correct_float)
    accuracy = w_num_correct/sum_weights
    return accuracy, w_num_correct, sum_weights

def test_model(generator, model):
    acc = 0.
    total_correct = 0
    total_pred = 0
    for b in generator:
        inputs, targets, ex_weight = b[0], b[1], b[2]
        pred = model(inputs)
        batch_acc, batch_correct, batch_pred = compute_accuracy(pred, targets, ex_weight)
        total_correct += batch_correct
        total_pred += batch_pred
    acc = total_correct/total_pred
    return acc

def predictUserInput(tweet, NNmodel, alay_dict, vocabulary, cm=False):
    if cm:
        twt = tweet
    else:
        twt = tc.text_preprocessing(tweet, alay_dict)
    inp = fastmath.numpy.array(NND.tweet2tensor(twt, vocabulary=vocabulary))
    inp = inp[None, :]
    pred_prob = NNmodel(inp)
    pred = int(pred_prob[0,1] > pred_prob[0,0])
    sentiment = 'negative'
    if pred == 1: sentiment='positive'
    return pred, sentiment

def confusionMatrix(x, y, model, alay_dict, vocabulary):
    y_true, y_pred = [], []

    for i,n in zip(x,y):
        try:
            pred, sent = predictUserInput(i, model, alay_dict, vocabulary, True)
            y_pred.append(pred)
            y_true.append(n)
        except:
            continue

    return classification_report(y_true, y_pred)