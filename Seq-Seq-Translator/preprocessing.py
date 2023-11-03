import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy.datasets import Multi30k
# from torchtext.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

# from torchtext.data import Field , BucketIterator
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter
from utils import translate_sentence , bleu , save_checkpoint, load_checkpoint

root_dir = ".data"

spacy_ger = spacy.load('de_core_news_sm')
spacy_eng = spacy.load('en_core_web_sm')

def tokenizer_ger(text):

    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenizer_eng(text):

    return [tok.text for tok in spacy_eng.tokenizer(text)]


english = Field(sequential=True, use_vocab=True, tokenize= tokenizer_eng, lower=True)

german = Field(sequential=True, use_vocab=True, tokenize= tokenizer_ger, lower=True)


train_data, validation_data, test_data = Multi30k.splits(exts=('.de','.en'),
                                                         fields=(german, english))


english.build_vocab(train_data, max_size=10000, min_freq=2)
german.build_vocab(train_data, max_size=10000, min_freq=2)
