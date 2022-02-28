import pandas as pd
#from .preprocessing import batchify_docs, Corpus
#
from .bert_crossling_prep import batchify_docs, Corpus   
from .keyword_extraction_main import predict
import langid                 


def extract_keywords(text, model, dictionary, sp, stemmer, args):
    all_docs = [[1, text, langid.classify(text)[0],"nema"]]
    df_test = pd.DataFrame(all_docs)
    df_test.columns = ["id", "text","lang","keyword"]

    #corpus = Corpus(df_test, dictionary, sp, args)
    corpus = Corpus(df_test, df_test, df_test, args)
    test_data, targets , langs = batchify_docs(corpus.test, corpus.test_target, corpus.test_langs, 1)

    model.eval()
    predictions = predict(test_data, model, sp, corpus, args, langs, targets)
    predictions_lemmas = []
    for kw in predictions:
        if len(kw.split()) == 1:
            lemma_kw = stemmer(kw)
        else:
            lemma_kw = kw
        predictions_lemmas.append(lemma_kw)
    root_kws = {}
    for idx, lemma in enumerate(predictions_lemmas):
        if not lemma in root_kws:
            root_kws[lemma] = predictions[idx]
    present = []
    for idx, lemma in enumerate(predictions_lemmas):
        kw = root_kws[lemma] 
        present.append(kw)
                
    return present[:args['kw_cut']]


