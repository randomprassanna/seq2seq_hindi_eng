import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torch.utils.tensorboard import SummaryWriter

from torchtext.vocab import vocab
from collections import Counter, OrderedDict
from torchtext.vocab import Vocab
from torch.utils.data import DataLoader

small_data = pd.read_csv('small_data.csv')
train, testtemp = train_test_split(small_data, test_size=0.3, shuffle=True)
val, test = train_test_split(testtemp, test_size=0.5, shuffle=True)

spacy_eng = spacy.load("en_core_web_sm")
spacy_hi = spacy.load("xx_sent_ud_sm") #multilang model

def tokenize_hi(text):
    return [tok.text for tok in spacy_hi.tokenizer(text)]

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


hindi = train.iloc[:,1].values
eng = train.iloc[:,2].values

tok_hi =[(tokenize_hi(sent)) for sent in hindi]
tok_eng =[(tokenize_eng(sent)) for sent in eng]

tok_lang = tok_eng
def get_counter(tok_lang):
    counter_lang = Counter()
    for i in range(len(tok_lang)):
        counter_lang.update(tok_lang[i])
    return counter_lang


def build_vocab_with_spl(tok_lang):
    counter_lang = get_counter(tok_lang)
    sorted_by_freq_tuples_lang = sorted(counter_lang.items(), key=lambda x: x[1], reverse=True)
    # text as a key,index as a value
    ordered_dict_lang = OrderedDict(sorted_by_freq_tuples_lang)
    voc_lang = vocab(ordered_dict_lang, min_freq=1)
    spl_tok = ['<pad>', '<bos>', '<eos>', '<unk>']
    spl_tok_idx = [0, 1, 2, 3]
    for (spl_tok, spl_tok_idx) in zip(spl_tok, spl_tok_idx):
        voc_lang.insert_token(spl_tok, spl_tok_idx)

    default_index = voc_lang['<unk>']
    voc_lang.set_default_index(default_index)
    return voc_lang

voc_eng = build_vocab_with_spl(tok_eng)
voc_hi = build_vocab_with_spl(tok_hi)
print(len(voc_hi))


def get_sent_index_array(sent, voc_lang):
    arr_lang = [voc_lang([word]) for word in sent]
    return arr_lang, len(arr_lang)


def get_seq_and_len_and_join_spl(data_lang, voc_lang, tokenize_lang):
    list_seq_tensor_lang = []
    len_seq_list_lang = []
    for_max = []
    for sent in data_lang:
        arr, length = get_sent_index_array(tokenize_lang(sent), voc_lang)
        list_seq_tensor_lang.append(torch.tensor(arr))
        len_seq_list_lang.append(torch.tensor(length))
        for_max.append(length)
        ## as we will add 2 tokens to evry sentence
        max_sequence = torch.LongTensor(for_max).max() + 2

    BOS_IDX = voc_lang['<bos>']
    EOS_IDX = voc_lang['<eos>']
    print(len(list_seq_tensor_lang))
    for i in range(len(list_seq_tensor_lang)):
        templist = []
        list_seq_tensor_lang[i] = torch.cat([torch.LongTensor([[BOS_IDX]]),
                                             list_seq_tensor_lang[i], torch.LongTensor([[EOS_IDX]])], dim=0)

    list_seq = list_seq_tensor_lang

    return list_seq, len_seq_list_lang, max_sequence


def pad_sequences(data_lang, voc_lang, tokenize_lang):
    list_seq, len_seq_list_lang, max_sequence = get_seq_and_len_and_join_spl(data_lang, voc_lang, tokenize_lang)
    flist_seq_tensors_lang = []

    for (idx, (seq, len_seq)) in enumerate(zip(list_seq, len_seq_list_lang)):
        seq_tensor = torch.zeros((1, max_sequence))
        seq_tensor[0, :len_seq + 2] = seq.squeeze(1)
        flist_seq_tensors_lang.append(seq_tensor)
    return flist_seq_tensors_lang


def make_variables_lang(data_lang, voc_lang, tokenize_lang):
    return pad_sequences(data_lang, voc_lang, tokenize_lang)


def createmydataset(data_lang1, data_lang2, voc_lang1, voc_lang2, tokenize_lang1, tokenize_lang2):
    which_dataset = []
    list_lang1_tensors = make_variables_lang(data_lang1, voc_lang1, tokenize_lang1)
    list_lang2_tensors = make_variables_lang(data_lang1, voc_lang2, tokenize_lang2)

    for (lang1_tensors, lang2_tensors) in zip(list_lang1_tensors, list_lang2_tensors):
        which_dataset.append((lang1_tensors.squeeze(), lang2_tensors.squeeze()))
        ## we have to squeeze so we will get 1 dimensional tensor
    return which_dataset

ds = createmydataset(eng,hindi,voc_eng,voc_hi,tokenize_eng,tokenize_hi)
print(len(ds))

print(ds[56][0].shape)
print(type(ds[76]))


batch_size = 3
num_workers = 2

def get_loader(dataset,batch_size,num_workers):
    dl = DataLoader(ds, batch_size=batch_size,
                          shuffle=True, num_workers=num_workers, pin_memory=True)
    return dl

train_dl = get_loader(ds,batch_size, num_workers)

for x,y in train_dl:
    print(y.shape)
    print(x.shape)
    break

print(x.dtype)

