import io
import time
import multiprocessing
import pandas as pd
import numpy as np
from datetime import timedelta
import gensim
from gensim.models import word2vec, Word2Vec

def GenerateWikiData():
    start_time = time.time()
    print('Streaming Wiki corpus Bahasa...')
    id_wiki = gensim.corpora.WikiCorpus('EngineFiles/Word2Vec/idwiki-latest-pages-articles.xml.bz2', lemmatize=False, dictionary={}, lower=True)
    article_count = 0

    with io.open('EngineFiles/Word2Vec/idwiki.txt', 'w', encoding='utf-8') as wiki_txt:
        for i in id_wiki.get_texts():
            wiki_txt.write(' '.join(map(str, i)) + '\n')
            article_count += 1

            if article_count % 10000 == 0:
                print(f'{article_count} articles processed')

        print(f'Total : {article_count} articles')

    finish_time = time.time()
    print('Elapsed time: {}'.format(timedelta(seconds=finish_time-start_time)))

def word_vector(tokens, size, model):
    vec = np.zeros(size).reshape((1,size))
    count = 0

    for w in tokens:
        try:
            vec += model[w].reshape((1,size))
            count += 1
        except KeyError:
            continue

    if count != 0:
        vec /= count

    return vec

def vectorize_tweet(word2vec_model, dataframe=pd.DataFrame(), tweet_column=''):
    tt = dataframe[tweet_column].apply(lambda x : x.split())
    wordvec_arrays = np.zeros((len(tt), 200))

    for i in range(len(tt)):
        wordvec_arrays[i,:] = word_vector(tt[i], 200, word2vec_model)

    wordvec_df = pd.DataFrame(wordvec_arrays)
    return wordvec_df

def TrainModel(dataframe=pd.DataFrame(), tweet_column='', model_name='model'):
    # Tokenizing all tweets
    # tt = dataframe[tweet_column].apply(lambda x : x.split())

    start_time = time.time()
    print('Training Word2Vec model...')
    sentences = word2vec.LineSentence('EngineFiles/Word2Vec/idwiki.txt')

    # Train the model
    id_w2v = word2vec.Word2Vec(sentences, size=200, workers=multiprocessing.cpu_count()-1, window=5)
    # id_w2v.train(tt, total_examples=len(dataframe[tweet_column]), epochs=20)

    # Export the model
    id_w2v.save('EngineFiles/Word2Vec/model/'+model_name+'.model')
    finish_time = time.time()
    print('Finished. Elapsed time : {}'.format(timedelta(seconds=finish_time-start_time)))
    print('\nModel is saved in EngineFiles/Word2Vec/model/'+model_name+'.model directory')

def GetModel(model_name=''):
    try:
        model = Word2Vec.load('EngineFiles/Word2Vec/model/'+model_name+'.model')
        return model
    except:
        'model cannot be found!'