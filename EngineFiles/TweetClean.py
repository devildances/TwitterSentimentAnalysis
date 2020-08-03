import pandas as pd
import numpy as np
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re

def get_low(sentence):
    return sentence.lower()

def del_punc(sentence):
    return ''.join([w for w in sentence if w not in string.punctuation])

def del_dirty_words(sentence):
    sentence = re.sub('\n', ' ', sentence)
    sentence = re.sub('\t', ' ', sentence)
    printable = set(string.printable)
    sentence = ''.join(filter(lambda x : x in printable, sentence))
    sentence = re.sub('user/W', ' ', sentence)
    sentence = re.sub('url', ' ', sentence)
    sentence = re.sub('http', ' ', sentence)
    sentence = re.sub('ssl', ' ', sentence)
    sentence = re.sub('btw', ' ', sentence)
    sentence = re.sub('(@[a-z0-9]+)\w+', ' ', sentence)
    sentence = re.sub('<[^>]*>', '', sentence)
    sentence = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', ' ', sentence)
    sentence = re.sub('[^0-9a-zA-Z]+', ' ', sentence)
    sentence = re.sub('x[*0-9a-zA-Z]+', '', sentence)
    sentence = re.sub('x[*0-9]+', '', sentence)
    sentence = re.sub('\d+', '', sentence)
    sentence = re.sub('  +', ' ', sentence)
    return sentence

def replace_alay(sentence, alay_dictionary):
    alay_dict = dict(zip(alay_dictionary['original'], alay_dictionary['replacement']))
    return ' '.join([alay_dict[w] if w in alay_dict else w for w in sentence.split()])

def stemming_words(sentence):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return stemmer.stem(sentence)

def del_stopwords(sentence, stopwords_list):
    sentence = ' '.join(['' if w in stopwords_list else w for w in sentence.split()])
    sentence = re.sub('  +', ' ', sentence)
    sentence = sentence.strip()
    return sentence

def text_preprocessing(sentence, alay_dictionary, stopwords_list):
    sentence = get_low(sentence)
    sentence = del_punc(sentence)
    sentence = del_dirty_words(sentence)
    sentence = replace_alay(sentence, alay_dictionary)
    sentence = stemming_words(sentence)
    sentence = del_stopwords(sentence, stopwords_list)
    return sentence