import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import defaultdict

mpl.style.use('seaborn')

def wordcloud_viz(dataframe=pd.DataFrame(), col_labels='', labels=[], col_target='', max_words=10):
    fig, ax = plt.subplots(1,len(labels), figsize=(len(labels)*10, len(labels)*10))
    for i in range(len(labels)):
        labls = ' '.join(dataframe[dataframe[col_labels]==labels[i]][col_target].tolist())
        ax[i].imshow(WordCloud(max_font_size=50, max_words=max_words, background_color='white').generate(labls), interpolation='bilinear')
        ax[i].set_title(labels[i].upper()+' tweets', fontsize=30)
        ax[i].axis('off')
    plt.show()

def generate_ngrams(text, stopwords_list, n_gram=1):
    token = [token for token in text.lower().split() if token != '' if token not in stopwords_list]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [' '.join(ngram) for ngram in ngrams]

def ngrams_viz(dataframe=pd.DataFrame(), col_labels='', labels=[], col_target='', ngrams=1, max_words=1, stopwords_list=[]):
    fig, ax = plt.subplots(ncols=len(labels), figsize=(27, 50), dpi=100)
    plt.tight_layout()
    c = ['red', 'green']
    gr = {1:'unigrams', 2:'bigrams', 3:'trigrams'}

    for i in range(len(labels)):
        grams = defaultdict(int)

        for t in dataframe[dataframe[col_labels]==labels[i]][col_target]:
            for w in generate_ngrams(text=t,stopwords_list=stopwords_list,n_gram=ngrams):
                grams[w] += 1

        df_grams = pd.DataFrame(sorted(grams.items(), key=lambda x : x[1])[::-1])
        sns.barplot(y=df_grams[0].values[:max_words], x=df_grams[1].values[:max_words], ax=ax[i], color=c[i])
        ax[i].spines['right'].set_visible(False)
        ax[i].set_xlabel('')
        ax[i].set_ylabel('')
        ax[i].tick_params(axis='x', labelsize=12)
        ax[i].tick_params(axis='y', labelsize=12)
        ax[i].set_title('Top 100 most common '+gr[ngrams]+' in '+labels[i].upper()+' tweets', fontsize=14)

    plt.show()

def words_and_chars_in_tweet(dataframe=pd.DataFrame(), col_labels='', labels=[], target='', color=[], generate=''):
    if generate.lower() == 'char':
        fig, ax = plt.subplots(1, len(labels), figsize=(len(labels)*7,10))
        for i in range(len(labels)):
            tweet_len = dataframe[dataframe[col_labels]==labels[i]][target].str.len()
            ax[i].hist(tweet_len, color=color[i])
            ax[i].set_title(labels[i].upper()+' Tweets Length')
            ax[i].grid(b=True, which='major', color='#666666', linestyle='-')
        fig.suptitle('Total Characters for Each Tweet')
        plt.show()
    elif generate.lower() == 'word':
        fig, ax = plt.subplots(1, len(labels), figsize=(len(labels)*7,10))
        for i in range(len(labels)):
            tweet_len = dataframe[dataframe[col_labels]==labels[i]][target].str.split().map(lambda x : len(x))
            ax[i].hist(tweet_len, color=color[i])
            ax[i].set_title(labels[i].upper()+' Tweets Words')
            ax[i].grid(b=True, which='major', color='#666666', linestyle='-')
        fig.suptitle('Total Words for Each Tweet')
        plt.show()
    elif generate.lower() == 'avg':
        fig, ax = plt.subplots(1, len(labels), figsize=(len(labels)*7,10))
        for i in range(len(labels)):
            tweet = dataframe[dataframe[col_labels]==labels[i]][target].str.split().apply(lambda x : [len(n) for n in x])
            sns.distplot(tweet.map(lambda x : np.mean(x)), ax=ax[i], color=color[i])
            ax[i].set_title(labels[i].upper()+' Tweets')
            ax[i].grid(b=True, which='major', color='#666666', linestyle='-')
            ax[i].minorticks_on()
            ax[i].grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        fig.suptitle('Average word length in each tweet')
        plt.show()
    else:
        print('Nothing to show!')