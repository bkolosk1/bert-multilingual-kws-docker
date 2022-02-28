import torch
import sentencepiece as spm
import json
from nltk import sent_tokenize, word_tokenize
import math


class Dictionary(object):
    def __init__(self, max_vocab_size):
        self.word2idx = {}
        self.idx2word = []
        self.counter = {}
        self.total = 0
        self.max_vocab_size = max_vocab_size


    def add_word(self, word):
        self.counter.setdefault(word, 0)
        self.counter[word] += 1
        self.total += 1

    def sort_words(self, unk, pos_tags=[]):
        #print('Unknown token: ', unk)

        if unk:
            self.counter['<unk>'] = 10000000000
        self.counter['<mask>'] = 10000000001
        self.counter['<pad>'] = 10000000002
        freq_list = sorted(list(self.counter.items()), key=lambda x:x[1], reverse=True)
        print('Vocab size: ', len(freq_list))
        print('Most common words in vocab: ', freq_list[:100])
        print('Least common words in vocab: ', freq_list[-100:])
        if self.max_vocab_size > 0:
            freq_list = freq_list[:self.max_vocab_size]
        else:
            self.max_vocab_size = len(freq_list)
        for word, freq in freq_list:
            if word not in self.word2idx:
                self.idx2word.append(word)
                self.word2idx[word] = len(self.idx2word) - 1
        for word in pos_tags:
            if word not in self.word2idx:
                self.idx2word.append(word)
                self.word2idx[word] = len(self.idx2word) - 1


    def __len__(self):
        return len(self.idx2word)



class Corpus(object):
    def __init__(self, df_test, dictionary, sp, args):
        self.max_length = args['max_length']
        self.sp = sp
        self.dictionary = dictionary
        if not args['split_docs']:
            self.test, self.ids = self.tokenize_doc(df_test, max_length=self.max_length)
        else:
            self.test, self.ids = self.tokenize_doc_chunks(df_test, max_length=self.max_length)


    def preprocess_line(self, line):
        words = []

        id = line['id']
        text = line['text']
        text = text.replace('-', ' ')
        text = text.replace('/', ' ')
        text = text.replace('∗', ' ')
        for sent in sent_tokenize(text):
            sent = word_tokenize(sent)

            bpe_sent = []
            for w in sent:
                w = w.lower()
                bpe_word = self.sp.tokenize(w)
                bpe_sent.append(bpe_word)
                words.extend(bpe_word)
            words.append('<eos>')
        return words, id


    def tokenize_doc(self, df, max_length):
        ids = []

        x = torch.zeros([df.shape[0], max_length], dtype=torch.long)
        i = 0
        for _, line in df.iterrows():

            words, id = self.preprocess_line(line)
            ids.append(id)

            for j, word in enumerate(words):
                if word in self.dictionary.word2idx:
                    idx = self.dictionary.word2idx[word]
                else:
                    idx = self.dictionary.word2idx['<unk>']
                if j < max_length:
                    x[i][j] = idx
        return x, ids

    def tokenize_doc_chunks(self, df, max_length):
        ids = []
        
        for _, line in df.iterrows():

            words, id = self.preprocess_line(line)
            num_docs = math.ceil(len(words)/max_length)
            ids.append(id)
            x = torch.zeros([num_docs, max_length], dtype=torch.long)
            i = 0
            j = 0

            for word in words:
                if word in self.dictionary.word2idx:
                    idx = self.dictionary.word2idx[word]
                else:
                    idx = self.dictionary.word2idx['<unk>']
                if j!=0 and j % max_length == 0:
                    i += 1
                    j = 0
                x[i][j] = idx
                j += 1
        return x, ids




def batchify_docs(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    doc_length = data.size(1)
    data = data.narrow(0, 0, nbatch * bsz)

    #Evenly divide the data across the bsz batches.
    data = data.view(-1, bsz, doc_length).contiguous()

    #print(data.size(), targets.size())
    return data


def get_batch_docs(data, i, cuda):
    if cuda:
        return data[i, :, :].cuda()
    return data[i, :, :]