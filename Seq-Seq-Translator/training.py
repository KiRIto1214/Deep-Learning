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

from model import Seq2Seq,Encoder,Decoder
from preprocessing import german,english,train_data,validation_data,test_data

#hyper parameters

num_epoch = 10
learning_rate = 0.001
batch_size = 64



#model parameters

load_model = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)


output_size = len(english.vocab)

encoder_embedding_size = 300
decoder_embedding_size = 300

hidden_size = 1024

num_layers = 2

enc_dropout = 0.5

dec_dropout = 0.5



writer = SummaryWriter(f'runs/loss_plot')

step = 0
 



train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
    (train_data,validation_data,test_data),
    batch_size=64,
    device='cuda' , sort_within_batch = True ,
    sort_key = lambda x : len(x.src)
)

encoder_net = Encoder(input_size_encoder,encoder_embedding_size,
                      hidden_size,num_layers, enc_dropout).to(device)


decoder_net = Decoder(input_size_decoder,decoder_embedding_size,
                    hidden_size,num_layers,dec_dropout).to(device)


model = Seq2Seq(encoder_net , decoder_net).to(device)

pad_idx = english.vocab.stoi['<pad>']

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(),criterion=criterion , lr=learning_rate)

if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.ptar'),model ,optimizer)


for epoch in range(num_epoch):
    print(f'Epoch [{epoch} / {num_epoch}]')
    
    checkpoint = {'state_dict': model.state_dict(), 'optimizer':optimizer.state_dict()}

    save_checkpoint(checkpoint)

    model.eval()

    # translate_sentence = translate_sentence(model , sentence, german
    # english , device , max_length=50)

    # print(f'Translated example\n {translate_sentence}')

    model.train()

    for batch_idx , batch in enumerate(train_iterator):

        input_data = batch.src.to(device)

        target = batch.trg.to(device)

        #output shape : (target_len , batch_size , output_dim) 

        output  = output[1:].reshape(-1 , output.shape[2])

        target = target[1:].reshape(-1)

        optimizer.zero_grad()

        loss = criterion(output, target)

        loss.backward()


        torch.nn.utlis.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()

        writer.add_scaler('Training Loss' , loss , global_step = step)


        step += 1


score = bleu(test_data, model , german , english , device)

print(f'Bleu Score {score*100 }')