import numpy
import pandas as pd
from sklearn.model_selection import train_test_split

def dictFrequency(text, y):
    ys = numpy.squeeze(y).tolist()
    freq = {}

    for p, t in zip(ys, text):
        for w in t.split():
            pair = (w,p)
            if pair in freq:
                freq[pair] += 1
            else:
                freq[pair] = 1

    return freq

def splitSet(dataframe=pd.DataFrame(), test_size=0.2, rand_state=0):
    df = dataframe.copy()
    df['label'] = df['label'].apply(lambda x : 1 if x=='positive' else 0)
    x_tr, x_ts, y_tr, y_ts = train_test_split(df['Tweet'].values, df['label'].values, test_size=test_size, random_state=rand_state, stratify=df.label.values)
    y_tr.shape += (1,)
    y_ts.shape += (1,)

    return x_tr, x_ts, y_tr, y_ts

def sigmoid(z):
    return 1/(1 + numpy.exp(-z))

def gradientDescent(x, y, theta, alpha, num_iters):
    m = x.shape[0]

    for i in range(0, num_iters):
        z = numpy.dot(x, theta)
        h = sigmoid(z)
        J = -1. / m * ( numpy.dot(y.transpose(), numpy.log(h)) + numpy.dot((1-y).transpose(), numpy.log(1-h)) )
        theta = theta - (alpha/m) * numpy.dot(x.transpose(), (h-y))
        J = float(J)

        return J, theta

def extractFeatures(text, freqs):
    word = text.split()
    x = numpy.zeros((1,3))
    x[0,0] = 1

    for w in word:
        x[0,1] += freqs.get((w, 1.0), 0)
        x[0,2] += freqs.get((w, 0.0), 0)

    assert(x.shape == (1,3))

    return x

def LR_predictTweet(tweet, freqs, theta):
    '''
    Input:
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (3,1) vector of weights
    Output:
        y_pred: the probability of a tweet being positive or negative
    '''
    x = extractFeatures(tweet, freqs)
    y_pred = sigmoid(numpy.dot(x, theta))

    return y_pred

def logisticRegressionAccuracy(x, y, freqs, theta):
    """
    Input:
        x: a list of tweets
        y: (m, 1) vector with the corresponding labels for the list of tweets
        freqs: a dictionary with the frequency of each pair (or tuple)
        theta: weight vector of dimension (3, 1)
    Output:
        accuracy: (# of tweets classified correctly) / (total # of tweets)
    """
    y_preds = []

    for t in x:
        y_pred = LR_predictTweet(t, freqs, theta)
        if y_pred > 0.5:
            y_preds.append(1)
        else:
            y_preds.append(0)

    accuracy = (y_preds == numpy.squeeze(y)).sum() / len(x)

    return accuracy