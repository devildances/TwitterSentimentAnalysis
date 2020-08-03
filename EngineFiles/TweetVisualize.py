import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import defaultdict

def wordcloud_viz(dataframe=pd.DataFrame(), col_labels='', labels=[], col_target=''):
    fig, ax = plt.subplots(1,len(labels), figsize=(len(labels)*10, len(labels)*10))
    for i in range(len(labels)):
        labls = ' '.join(dataframe[dataframe[col_labels]==labels[i]][col_target].tolist())
        ax[i].imshow(WordCloud(max_font_size=50, max_words=300, background_color='white').generate(labls), interpolation='bilinear')
        ax[i].set_title(labels[i].upper()+' tweets', fontsize=30)
        ax[i].axis('off')
    plt.show()

def generate_ngrams(text, stopwords_list, n_gram=1):
    token = [token for token in text.lower().split() if token != '' if token not in stopwords_list]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [' '.join(ngram) for ngram in ngrams]

def ngrams_viz(dataframe=pd.DataFrame(), col_labels='', labels=[], col_target='', ngrams=1, stopwords_list=[]):
    fig, ax = plt.subplots(ncols=len(labels), figsize=(27, 50), dpi=100)
    plt.tight_layout()
    c = ['red', 'gold', 'green']
    gr = {1:'unigrams', 2:'bigrams', 3:'trigrams'}

    for i in range(len(labels)):
        grams = defaultdict(int)

        for t in dataframe[dataframe[col_labels]==labels[i]][col_target]:
            for w in generate_ngrams(text=t,stopwords_list=stopwords_list,n_gram=ngrams):
                grams[w] += 1

        df_grams = pd.DataFrame(sorted(grams.items(), key=lambda x : x[1])[::-1])
        sns.barplot(y=df_grams[0].values[:100], x=df_grams[1].values[:100], ax=ax[i], color=c[i])
        ax[i].spines['right'].set_visible(False)
        ax[i].set_xlabel('')
        ax[i].set_ylabel('')
        ax[i].tick_params(axis='x', labelsize=12)
        ax[i].tick_params(axis='y', labelsize=12)
        ax[i].set_title('Top 100 most common '+gr[ngrams]+' in '+labels[i].upper()+' tweets', fontsize=14)

    plt.show()