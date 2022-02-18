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


def sent_array(sent):
    arr = [voc_eng([word]) for word in sent]
    return arr, len(arr)


def get_seq_len(data):
    tok_data = [sent_array(tokenize_eng(sent)) for sent in data]
    seq_vec = [i[0] for i in tok_data]
    seq_len = torch.LongTensor([i[1] for i in tok_data])

    for idx, _ in enumerate(seq_vec):
        seq_vec[idx] = [[voc_eng['<bos>']]] + seq_vec[idx] + [[voc_eng['<eos>']]]

    return seq_vec, seq_len


seq_vec, seq_len=get_seq_len(eng)
seq_tensor = torch.zeros((len(seq_vec), seq_len.max()+2)).long()
# for idx, (seq_vec, seq_len) in enumerate(zip(seq_vec, seq_len)):
#     print(torch.LongTensor(seq_vec).squeeze(1))
#     break

# for idx, (seq_vec, seq_len) in enumerate(zip(seq_vec, seq_len)):
#     print(seq_tensor[idx, :seq_len+1])
#     break

for idx, (seq_vec, seq_len) in enumerate(zip(seq_vec, seq_len)):
    print(len(seq_vec))
    seq_tensor[idx,:seq_len+2] = torch.LongTensor(seq_vec).squeeze(1)

print(seq_tensor)
def pad_sequences(seq_vec, seq_len):
    seq_tensor = torch.zeros((len(seq_vec), seq_len.max()+2)).long()
    for idx, (seq_vec, seq_len) in enumerate(zip(seq_vec, seq_len)):
        print(len(seq_vec))
        seq_tensor[idx,:seq_len] = torch.LongTensor(seq_vec)

    return seq_tensor
#
# print(len(seq_vec))
# seq_tensor = pad_sequences(seq_vec, seq_len)
# print(seq_tensor)