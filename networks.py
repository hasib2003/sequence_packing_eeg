
import torch.nn as nn
import torch
import math 
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 320):
        
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        print("pos_embedding.shape ",pos_embedding.shape)
        pos_embedding = pos_embedding.unsqueeze(-2)
        print("pos_embedding.shape ",pos_embedding.shape)
        

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):

        # assuming the input is in batch_first fashion batch_size, seqlen,features
        
        encodings = torch.squeeze(self.pos_embedding[:token_embedding.size(1), :])


        return self.dropout(token_embedding + encodings)
    
### the architecture is inspired from https://www.nature.com/articles/s41598-022-18502-3
class Temporal_Encoder(nn.Module): 
    def __init__(self,
                 num_heads = 8,
                 num_layers = 4,
                 input_features=64 # number of channels being used
                 ):

        
        super(Temporal_Encoder, self).__init__()
        
        self.num_heads = num_heads

        self.enc_layer = nn.TransformerEncoderLayer(d_model=input_features, nhead=num_heads,batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.enc_layer,num_layers=num_layers)

    def forward(self, x,src_mask):


        # x should be shape : batch, signal_length,num_channels
        return self.transformer_encoder(x,mask=src_mask)
    
##################################################################################
# used to merge the outputs of spatial and temporal encoders

class Classification_Head(nn.Module): 
    def __init__(self,dim_encoder=64,num_classes=109):
        
        
        super(Classification_Head, self).__init__()
        
        self.fc1 = nn.Linear(in_features=dim_encoder,out_features=num_classes)


        
    def forward(self, x):

        return self.fc1(x)    
    
##################################################################################
