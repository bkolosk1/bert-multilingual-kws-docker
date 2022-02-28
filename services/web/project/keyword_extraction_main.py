import argparse
import torch
import sentencepiece as spm
import torch.nn.functional as F
from .bert_crossling_prep import get_batch_docs
import sys
from . import model
from . import preprocessing
import pickle
from . import bert_crossling_prep# import get_batch, Corpus, batchify, batchify_docs, get_batch_docs, file_to_df



def loadDict(dict_path):
    sys.modules['preprocessing'] = preprocessing
    sys.modules['bert_crossling_prep'] = bert_crossling_prep

    with open(dict_path, 'rb') as file:
        kw_dictionary = pickle.load(file)
    return kw_dictionary


def loadModel(model_path, cuda):
    sys.modules['model'] = model
    sys.modules['bert_crossling_prep'] = bert_crossling_prep
    
    if not cuda:
        kw_model = torch.load(model_path, map_location=torch.device('cpu'))#, return_dict=False)
    else:
        kw_model = torch.load(model_path)
    kw_model.config.cuda = cuda
    if cuda:
        kw_model.cuda()
    else:
        kw_model.cpu()
    return kw_model


def get_tagset(path):
    tagset = {}
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.lower() in tagset:
                if any(letter.isupper() for letter in line):
                    tagset[line.lower()] = line
            else:
                tagset[line.lower()] = line
    return tagset

def predict(test_data, model, sp, corpus, args, langs, targets):

    step = 1
    cut = 0
    all_steps = test_data.size(0)
    encoder_pos = None
    total_pred = []
    stemmer = args['stemmer']

    with torch.no_grad():


        all_predicted_save = []
        all_batch_langs = []
        
        for i in range(0, all_steps - cut, step):
                    encoder_words, batch_labels, batch_langs = get_batch_docs(test_data, targets, langs, i, args['cuda'])
                    input_batch_labels = batch_labels.clone()
                    #encoder_words = get_batch_docs(test_data, i, args['cuda'])
                    loss, logits = model(encoder_words, input_pos=encoder_pos, lm_labels=input_batch_labels, test=True)#, return_dict=False)

                    maxes = []

                    batch_counter = 0
                    all_batch_langs = batch_langs.cpu().numpy()

                    for batch_idx, batch in enumerate(logits):
                                #Preds: ", crf_preds[batch_idx])
                                #print("True: ", input_batch_labels[batch_idx].cpu().numpy().tolist())
                                #print('---------------------------------------------------------------')
                                curr_lang_idx = all_batch_langs[batch_idx,0]
                               # stemmer = idx2stemmer[curr_lang_idx]
                                pred_save = []

                                pred_example = []
                                batch = F.softmax(batch, dim=1)
                                #print("predictions batch: ", batch.max(1)[1])
                                length = batch.size(0)
                                position = 0

                                pred_vector = []
                                probs_dict = {}
                                while position < len(batch):
                                    pred = batch[position]
                                    if not args['crf']:
                                        _ , idx = pred.max(0)
                                        idx = idx.item()                                    

                                    pred_vector.append(pred)
                                    pred_word = []

                                    if idx == 1:
                                        words = []
                                        num_steps = length - position
                                        for j in range(num_steps):
                                            new_pred = batch[position + j]
                                            values, new_idx = new_pred.max(0)

                                            if not args['crf']:
                                                new_idx = new_idx.item()
                                          
                                            prob = values.item()

                                            if new_idx == 1:
                                                word = corpus.dictionary.idx2word[encoder_words[batch_counter][position + j].item()]
                                                words.append((word, prob))
                                                pred_word.append((word, prob))

                                                #add max word prob in document to prob dictionary
                                                stem = stemmer(word)

                                                if stem not in probs_dict:
                                                    probs_dict[stem] = prob
                                                else:
                                                    if probs_dict[stem] < prob:
                                                        probs_dict[stem] = prob
                                            else:
                                                if sp is not None:
                                                    word = corpus.dictionary.idx2word[encoder_words[batch_counter][position + j].item()]
                                                    #if not word.startswith('Ġ'):
                                                    if word.startswith('##'):
                                                        words.append((word, prob))
                                                        stem = stemmer(word)

                                                        if stem not in probs_dict:
                                                            probs_dict[stem] = prob
                                                        else:
                                                            if probs_dict[stem] < prob:
                                                                probs_dict[stem] = prob
                                                break

                                        position += j + 1
                                        words = [x[0] for x in words]
                                        pred_example.append(words)
                                        pred_save.append(pred_word)
                                    else:
                                        position += 1

                                all_predicted_save.append(pred_save)

                                #assign probabilities
                                pred_examples_with_probs = []
                                #print("Candidate: ", pred_example)
                                for kw in pred_example:
                                    probs = []
                                    
                                    for word in kw:
                                        stem = stemmer(word)

                                        probs.append(probs_dict[stem])

                                    kw_prob = sum(probs)/len(probs)
                                    pred_examples_with_probs.append((" ".join(kw), kw_prob))

                                pred_example = pred_examples_with_probs

                                #sort by softmax probability
                                pred_example = sorted(pred_example, reverse=True, key=lambda x: x[1])

                                #remove keywords that contain punctuation and duplicates
                                all_kw = set()
                                filtered_pred_example = []
                                kw_stems = []

                                punctuation = "!$%&'()*+,.:;<=>?@[\]^_`{|}~"

                
                                for kw, prob in pred_example:                        
                                    kw_decoded = kw.replace(' ##', '')
                                    kw_stem = " ".join([stemmer(word) for word in kw_decoded.split()])
                                    kw_stems.append(kw_stem)

                                    if kw_stem not in all_kw and len(kw_stem.split()) == len(set(kw_stem.split())):
                                        has_punct = False
                                        for punct in punctuation:
                                            if punct in kw:
                                                has_punct = True
                                                break
                                        kw_decoded = kw.replace(' ##', '')
                                        if not has_punct and len(kw_decoded.split()) < 5:
                                            filtered_pred_example.append((kw, prob))
                                    all_kw.add(kw_stem)

                                pred_example = filtered_pred_example
                                filtered_pred_example = [x[0] for x in pred_example][:args['kw_cut']]

                                maxes.append(filtered_pred_example)
                                batch_counter += 1

                                if sp is not None:
                                    all_decoded_maxes = []
                                    for doc in maxes:
                                        decoded_maxes = []
                                        for kw in doc:
                                            kw = kw.replace(' ##', '')
                                            decoded_maxes.append(kw)
                                        all_decoded_maxes.append(decoded_maxes)

                                    maxes = all_decoded_maxes

                                total_pred.extend(maxes)
    return total_pred[0]


