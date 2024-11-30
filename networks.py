
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
                 num_heads = 7,
                 num_layers = 4,
                 input_features=14 # number of channels being used
                 ):

        
        super(Temporal_Encoder, self).__init__()
        
        self.num_heads = num_heads

        self.enc_layer = nn.TransformerEncoderLayer(d_model=input_features, nhead=num_heads,batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.enc_layer,num_layers=num_layers)

    def forward(self, x,src_mask=None):


        # x should be shape : batch, signal_length,num_channels

        if src_mask is not None:
            return self.transformer_encoder(x,mask=src_mask)
        
        return self.transformer_encoder(x)
    

class Spatial_Encoder(nn.Module): 
    def __init__(self,
                 num_heads = 8,
                 num_layers = 4,
                 input_features=32 # number of channels being used
                 ):

        
        super(Spatial_Encoder, self).__init__()
        
        self.num_heads = num_heads

        self.enc_layer = nn.TransformerEncoderLayer(d_model=input_features, nhead=num_heads,batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.enc_layer,num_layers=num_layers)

    def forward(self, x,src_mask=None):

        x = torch.permute(x,(0,2,1))


        # x should be shape : batch, signal_length,num_channels

        if src_mask is not None:
            return self.transformer_encoder(x,mask=src_mask)
        
        return self.transformer_encoder(x)
        

    
##################################################################################


# used to merge the outputs of spatial and temporal encoders

class DenseNet(nn.Module): 
    def __init__(self,inputFeature=448,projDim=128):
        
        
        super(DenseNet, self).__init__()
        
        self.fc1 = nn.Linear(in_features=inputFeature,out_features=256)
        self.fc2 = nn.Linear(in_features=256,out_features=projDim)


        
    def forward(self, x):

        x = torch.flatten(x,start_dim=1)
        
        x = self.fc1(x)
        x = self.fc2(x)

        return x
    

class ProjFormer(nn.Sequential):
    
    def __init__(self, num_channel=14,seq_len=32):
        super().__init__(

            Temporal_Encoder(input_features=num_channel),
            Spatial_Encoder(input_features=seq_len),
            DenseNet(inputFeature=num_channel*seq_len)
        )

    
##################################################################################
