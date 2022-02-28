from lemmagen3 import Lemmatizer
import string
import pandas as pd
from nltk.tokenize import word_tokenize
import random
import pickle
from stopwordsiso import stopwords
import os
import LatvianStemmer

def get_lemmatizer(lang):
    lemmatizer = -1
    if lang == 'ee':
        lemmatizer = Lemmatizer('et').lemmatize
    if lang == 'ru':
        lemmatizer = Lemmatizer('ru').lemmatize
    if lang == 'lv':        
        lemmatizer = LatvianStemmer.stem
    assert not lemmatizer == -1
    return lemmatizer

def remove_punctuation(text):    
    table = text.maketrans({key: None for key in string.punctuation})
    text = text.translate(table)
    return text

def remove_stopwords(text, lang = 'et'):
    if lang == 'ee':
        lang = 'et'
    sw = stopwords(lang)
    for key in sw:
        text.replace(key,"")
    return text

def prepare(text, lang = 'et'):
        lemmatize = get_lemmatizer(lang)
        lowered = text.lower()
        no_stopwords = remove_stopwords(lowered)
        sentences = word_tokenize(no_stopwords, language='russian') if lang == 'ru' else word_tokenize(no_stopwords)
        cleaned = [remove_punctuation(sent) for sent in sentences]
        try: 
            sentence_lemmatized = ' '.join([lemmatize(token) for token in cleaned])
            return sentence_lemmatized
        except Exception as e:
            print(e)
            sentence_lemmatized = ' '.join([token for token in cleaned])
            return sentence_lemmatized
    
def build_kw(path, lang='et'):
    outs = {}
    if lang == 'rus':
        lang = 'ru'
    sw = stopwords(lang)
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip("\n")
            parsed = prepare(line, lang)        
            if not parsed in sw:
                outs[parsed] = outs.get(parsed,[]) + [line]
    return outs

def find_in_text(text, all_tags):
    found_tags = set()
    for tag in all_tags:
        if tag in text:
            found_tags.union(tag)
    return found_tags


def load(lang='ru'):
    with open(os.path.join("project","tf_idf_files", "tf_idf_"+lang+".pkl"), "rb") as f:
        tf_idf = pickle.load(f)
    with open(os.path.join("project","tf_idf_files", "keywords_"+lang+".pkl"), "rb") as f:       
        keywords_o = pickle.load(f)
    return keywords_o, tf_idf    

def save(keyword, tf_idf, lang='ru'):
    with open(os.path.join("project","tf_idf_files", "keywords_"+lang+".pkl"), "wb") as f:
        pickle.dump(keyword, f)
    with open(os.path.join("project","tf_idf_files", "tf_idf_"+lang+".pkl"), "wb") as f:
        pickle.dump(tf_idf, f)


def extract_kw(vec, features, lemma2kw, threshold = 0.0001, multi_strategy="random", n=10):
    """
    
    Parameters
    ----------
    vec : 
        vector with tf_idf values
        tf_idf of weights of the current text
    features : list(str)
        features from the tf_idf.
    threshold: float
        thresholding tf-idf values, from which on to ignore input values.
    lemma2kw : dict
        mapping of lemmas to original keywords.
    multi_strategy : str
        strategy how to pool values for lemmas covering multiple tags.
    n : TYPE, number of outputs
        the default is 10.

    Returns
    -------
    output : list of str
        returns

    """
    assert len(vec) == len(features)
    to_sort = []
    for x,y in zip(vec,features):
        if x >= threshold:
            to_sort.append((x,y))
    to_sort.sort(reverse=True)
    output = []
    if multi_strategy == "random":
        for prob, tag in to_sort[:n]:
            new_tag = ""
            possible_tags = lemma2kw[tag]
            m = len(possible_tags)  
            if m > 1 and multi_strategy == "random":
                new_tag = random.choice(possible_tags)
            elif m > 1 and multi_strategy == "min":
                new_tag = min(possible_tags, key=len)
            elif m > 1 and multi_strategy == "max":
                new_tag = max(possible_tags, key=len)
            else:
                new_tag = possible_tags
            output.append(new_tag)
    return output


def predict_tf_idf(text, lang='et', opt = 'nat'):
    key,tfidf_vectorizer = load(lang)
    parts = lang.split('_')
    lang = parts[0] if parts[1] == 'nat' else 'ru'
    prepared_text = prepare(text, lang)
    vectorized = tfidf_vectorizer.transform([prepared_text])
    get_kw = extract_kw(vectorized.T.todense(), tfidf_vectorizer.get_feature_names(), key)
    get_kw_o = []
    for x in get_kw:
     if type(list()) == type(x):
         for y in x:
             get_kw_o.append(y)
     else:
         get_kw_o.append(x)
    return get_kw_o
