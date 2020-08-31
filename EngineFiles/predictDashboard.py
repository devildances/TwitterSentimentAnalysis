import pandas as pd
import numpy
import os
# import sys
# sys.path.insert(1,'..')
from EngineFiles import TweetClean as tc
from EngineFiles import TweetFormat as tf
from EngineFiles.MachineLearning import LogisticRegressionModel as LR, NaiveBayesModel as NB
from EngineFiles.DeepLearning import NeuralNetworkDataPrepro as NND
from EngineFiles.DeepLearning import NeuralNetworkObject as NNO
from EngineFiles.DeepLearning import NeuralNetworkModel as NNM

def DashboardPredictionLoad():
    df = pd.read_csv('Data/indonesia_Tweet/clean_tweets.csv')
    df.dropna(subset=['Tweet'],inplace=True)
    alay_dict = tf.bahasa_slang()

    # LR model
    x_train, x_test, y_train, y_test = LR.splitSet(df, 0.2, 42)
    freq = LR.dictFrequency(x_train, y_train)
    X = numpy.zeros((len(x_train), 3))
    for i in range(len(x_train)):
        X[i, :] = LR.extractFeatures(x_train[i], freq)
    Y = y_train
    J, theta = LR.gradientDescent(X, Y, numpy.zeros((3, 1)), 1e-9, 10000)

    # NB model
    x_train, x_test, y_train, y_test = NB.splitSet(df, 0.2, 42)
    freqs = NB.countTweets(x_train, y_train)
    logprior, loglikelihood = NB.naiveBayesTrain(freqs, x_train, y_train)

    # NN model
    with open('EngineFiles/Word2Vec/idwiki_clean.txt', 'r', encoding='utf-8') as f:
        idwiki = f.read()
    idwiki = idwiki.split('\n')
    df_train, df_test, x_train, x_train_pos, x_train_neg, x_test, x_test_pos, x_test_neg, y_train, y_test, index_train, index_test = NND.splitDataset(df, 0.2, 42)
    vocab = NND.createVocab(x_train, idwiki)
    load_path = './EngineFiles/DeepLearning/model/'
    load_path = os.path.join(load_path, 'model.pkl.gz')
    load_mdl = NNM.classifier(vocab_size=len(vocab))
    load_mdl.init_from_file(load_path)

    return alay_dict, freq, theta, logprior, loglikelihood, vocab, load_mdl

def LR_DashboardPredictionResult(inputTweet, freq, theta, alay_dict):
    if len(inputTweet) >= 2 and type(inputTweet) == str:
        inputTweet = tc.text_preprocessing(inputTweet, alay_dict)
        result = LR.LR_predictTweet(inputTweet, freq, theta)
        if result < 0.5:
            result = 'negative'
        else:
            result = 'positive'
    else:
        result = 'Your input can\'t be empty or numbers only!'

    return result

def NB_DashboardPredictionResult(inputTweet, logprior, loglikelihood, alay_dict):
    if len(inputTweet) >= 2 and type(inputTweet) == str:
        inputTweet = tc.text_preprocessing(inputTweet, alay_dict)
        if NB.NB_predictTweet(inputTweet, logprior, loglikelihood) > 0:
            result = 'positive'
        else:
            result = 'negative'
    else:
        result = 'Your input can\'t be empty or numbers only!'

    return result

def NN_DashboardPredictionResult(inputTweet, NNmodel, alay_dict, vocabulary):
    if len(inputTweet) >= 2 and type(inputTweet) == str:
        pred, result = NNM.predictUserInput(inputTweet, NNmodel, alay_dict, vocabulary)
    else:
        result = 'Your input can\'t be empty or numbers only!'

    return result