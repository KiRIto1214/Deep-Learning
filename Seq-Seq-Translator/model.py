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

from preprocessing import english,german


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Encoder(nn.Module):
    
    def __init__(self, input_size , embedding_size, hidden_size , num_layers, p):

        super(Encoder,self).__init__()
        self.hidden_size = hidden_size

        self.num_layers = num_layers


        self.dropout = nn.Dropout(p)

        self.embedding = nn.Embedding(input_size , embedding_size)

        self.rnn = nn.LSTM(embedding_size,hidden_size, num_layers, dropout=p)


    def forward(self, x):

        '''so firstly a sentence will be tokenized 
        and will be mapped to some index corresponding to where it is in
         vocabulary and that vector is sending to the lstm 
         so x is vector is indices'''
        
        # x shape : (seq_length , N) where N is batch size

        embedding = self.dropout(self.embedding(x))
        #embedding : (seq_length , N , embedding_size)
        '''
        if embedding size=300 , so each word now has mapping to some
        300 dimensional space 
        '''

        outputs, (hidden, cell) = self.rnn(embedding)
        # we care about context vector that is hideen and cell


        return hidden ,cell


class Decoder(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size,output_size,
                 num_layers, dropout):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)

        self.embedding = nn.Embedding(input_size,embedding_size)

        self.rnn = nn.LSTM(embedding_size,hidden_size,dropout=p) 

        #hidden size of encoder and decoder is same

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x , hidden, cell):
        '''we gonna predict word by word,
            shape of x: (N) where N is batch size ; 
            but we want (1 , N) we can do it by using unsqueeze'''
        
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))
        # shape : (1, N , embedding_size)

        outputs , (hidden, cell) = self.rnn(embedding, (hidden,cell))

        #shape of outputs: (1 , N , hidden_size)

        predictions = self.fc(outputs)
        #shape of predictions = (1,N, length of vocab)

        predictions = predictions.squeeze(0)
        #shape (N , length of vocab)
        '''we are predicting one word at a time, but we want to predict all of
         the words in target sentence, so later we add the output of the decoder
           one step at a time so we have shape (N, length of vocab) '''

        return predictions,hidden,cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder) :
        super(Seq2Seq , self).__init__()

        self.encoder = encoder
        self.decoder = decoder


    def forward(self, source, target , teacher_force_ratio=0.5):

        batch_size = source.shape[1]
        # (target_len , N)

        target_len = target.shape[0]

        target_vocab_size = len(english.vocab)

        hidden, cell = self.encoder(source)

        outputs = torch.zeros(target_len, batch_size , target_vocab_size).to(device)

        '''we are gonna predict one word at a time '''


        #start token
        x = target[0]

        #word by word pred
        for t in range(1,target_len):

            output, hidden ,cell = self.decoder(x, hidden , cell)

            outputs[t] = output

            # (N , eng_vocab)
            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess
  
            

        return outputs

