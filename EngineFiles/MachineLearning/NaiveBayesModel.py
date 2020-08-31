import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from EngineFiles import TweetClean as tc
from EngineFiles import TweetFormat as tf

def countTweets(tweets, y):
    '''
    Input:
        tweets: a list of tweets
        y: a list corresponding to the sentiment of each tweet (either 0 or 1)
    Output:
        result: a dictionary mapping each pair to its frequency
    '''
    result = {}
    for i, tweet in zip(y, tweets):
        for w in tweet.split():
            pair = (w, i)
            if pair in result:
                result[pair] += 1
            else:
                result[pair] = 1

    return result

def splitSet(dataframe=pd.DataFrame(), test_size=0.2, rand_state=0):
    df = dataframe.copy()
    df['label'] = df['label'].apply(lambda x : 1 if x=='positive' else 0)
    x_tr, x_ts, y_tr, y_ts = train_test_split(df['Tweet'].values, df['label'].values, test_size=test_size, random_state=rand_state, stratify= df.label.values)

    return x_tr, x_ts, y_tr, y_ts

def lookup(freqs, word, label):
    '''
    Input:
        freqs: a dictionary with the frequency of each pair (or tuple)
        word: the word to look up
        label: the label corresponding to the word
    Output:
        n: the number of times the word with its corresponding label appears.
    '''
    n = 0
    pair = (word, label)
    if pair in freqs:
        n = freqs[pair]

    return n

def naiveBayesTrain(freqs, x, y):
    '''
    Input:
        freqs: dictionary from (word, label) to how often the word appears
        x: a list of tweets
        y: a list of labels correponding to the tweets (0,1)
    Output:
        logprior: the log prior. (equation 3 above)
        loglikelihood: the log likelihood of you Naive bayes equation. (equation 6 above)
    '''
    loglikelihood = {}
    vocab = set([p[0] for p in freqs.keys()])
    V = len(vocab)
    N_pos = N_neg = 0

    for p in freqs.keys():
        if p[1] > 0:
            N_pos += freqs[p]
        else:
            N_neg += freqs[p]

    D = len(y)
    D_pos = sum(y)
    D_neg = D - D_pos
    logprior = numpy.log(D_pos) - numpy.log(D_neg)

    for w in vocab:
        freq_pos = lookup(freqs, w, 1)
        freq_neg = lookup(freqs, w, 0)
        prob_w_pos = (freq_pos + 1)/(N_pos + V)
        prob_w_neg = (freq_neg + 1)/(N_neg + V)
        loglikelihood[w] = numpy.log(prob_w_pos/prob_w_neg)

    return logprior, loglikelihood

def NB_predictTweet(tweet, logprior, loglikelihood):
    '''
    Input:
        tweet: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Output:
        p: the sum of all the loglikelihoods of each word in the tweet (if found in the dictionary) + logprior (a number)
    '''
    word = tweet.split()
    p = logprior

    for w in word:
        if w in loglikelihood:
            p += loglikelihood[w]

    return p

def naiveBayesAccuracy(x, y, logprior, loglikelihood):
    """
    Input:
        x: A list of tweets
        y: the corresponding labels for the list of tweets
        logprior: the logprior
        loglikelihood: a dictionary with the loglikelihoods for each word
    Output:
        accuracy: (# of tweets classified correctly)/(total # of tweets)
    """
    y_preds = []

    for t in x:
        if NB_predictTweet(t, logprior, loglikelihood) > 0:
            y_pred = 1
        else:
            y_pred = 0
        y_preds.append(y_pred)

    error = numpy.mean(numpy.absolute(y_preds-y))
    accuracy = 1 - error

    return accuracy, y_preds