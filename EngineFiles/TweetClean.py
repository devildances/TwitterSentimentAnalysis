import pandas as pd
import numpy as np
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re
from nltk.corpus import stopwords

def get_low(sentence):
    return sentence.lower()

def del_punc(sentence):
    return ''.join([w for w in sentence if w not in string.punctuation])

def del_dirty_words(sentence):
    sentence = re.sub(r'\\n', ' ', sentence)
    sentence = re.sub(r'\\t', ' ', sentence)
    sentence = re.sub(r'\"', '', sentence)
    sentence = re.sub(r'\[username\]', '', sentence)
    sentence = re.sub(r'\[user\]', '', sentence)
    sentence = re.sub(r'\[url\]', '', sentence)
    sentence = re.sub(r'\\t', ' ', sentence)
    sentence = re.sub(r'@[A-Za-z0–9]+', '', sentence)
    sentence = re.sub(r'\$\w*', '', sentence)
    sentence = re.sub(r'(^(rt)|\s)+(rt)+\s', ' ', sentence)
    sentence = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', ' ', sentence)
    sentence = re.sub(r'#', '', sentence)
    sentence = re.sub(r'\\x[*0-9a-zA-Z]+', ' ', sentence)
    sentence = re.sub(r'user', '', sentence)
    sentence = re.sub(r'url', ' ', sentence)
    sentence = re.sub(r'ssl', ' ', sentence)
    sentence = re.sub(r'[^0-9a-zA-Z]+', ' ', sentence)
    sentence = re.sub(r'\d+', '', sentence)
    sentence = re.sub(r'  +', ' ', sentence)
    return sentence

def replace_alay(sentence, alay_dictionary):
    alay_dict = dict(zip(alay_dictionary['original'], alay_dictionary['replacement']))
    return ' '.join([alay_dict[w] if w in alay_dict else w for w in sentence.split()])

def stemming_words(sentence):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return stemmer.stem(sentence)

def del_stopwords(sentence):
    stopwords_list = stopwords.words('indonesian')
    sentence = ' '.join(['' if w in stopwords_list else w for w in sentence.split()])
    sentence = re.sub('  +', ' ', sentence)
    sentence = sentence.strip()
    return sentence

def text_preprocessing(sentence, alay_dictionary):
    sentence = get_low(sentence)
    sentence = del_dirty_words(sentence)
    sentence = del_punc(sentence)
    sentence = replace_alay(sentence, alay_dictionary)
    sentence = stemming_words(sentence)
    sentence = del_stopwords(sentence)
    return sentence
