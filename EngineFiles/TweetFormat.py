import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

def main_data():
    df = pd.read_csv('Data/indonesia_tweet/data.csv', encoding='latin-1')
    df['label'] = 'nan'
    for i, v in df.transpose().items():
        if (v['HS']==0) & (v['Abusive']==0) & (v['HS_Individual']==0) & (v['HS_Group']==0) & (v['HS_Religion']==0) & (v['HS_Race']==0) &\
        (v['HS_Physical']==0) & (v['HS_Gender']==0) & (v['HS_Other']==0) & (v['HS_Weak']==0) & (v['HS_Moderate']==0) & (v['HS_Strong']==0):
            df['label'][i] = 'normal'
        elif (v['HS']==0) & (v['Abusive']==1) & (v['HS_Individual']==0) & (v['HS_Group']==0) & (v['HS_Religion']==0) & (v['HS_Race']==0) &\
        (v['HS_Physical']==0) & (v['HS_Gender']==0) & (v['HS_Other']==0) & (v['HS_Weak']==0) & (v['HS_Moderate']==0) & (v['HS_Strong']==0):
            df['label'][i] = 'abusive'
        else:
            df['label'][i] = 'hatespeech'
    df = df[['Tweet','label']]
    return df

def bahasa_slang():
    slang_a = pd.read_csv('Data/indonesia_tweet/new_kamusalay.csv', encoding='latin-1', names=['original', 'replacement'])
    slang_b = pd.read_csv('Data/indonesia_tweet/kamusalay.csv', usecols=['slang', 'formal'])
    slang_b.columns = ['original', 'replacement']
    slang_b.drop_duplicates(subset=['original'], keep='first', inplace=True, ignore_index=True)
    slang = pd.concat([slang_a, slang_b], ignore_index=True)
    slang.drop_duplicates(subset=['original'], keep='first', inplace=True, ignore_index=True)
    return slang

def bahasa_stopwords(additional_words=[]):
    stopword_a = pd.read_csv('Data/indonesia_tweet/stopwordbahasa.csv', names=['stopword'])
    stopword = stopword_a['stopword'].tolist()
    factory = StopWordRemoverFactory()
    stopword_b = factory.get_stop_words()
    stopword.extend(stopword_b)
    if type(additional_words) != list:
        return TypeError
    else:
        if len(additional_words) > 0:
            stopword.extend(additional_words)
    stopword = list(dict.fromkeys(stopword))
    return stopword