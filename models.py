#### contains the definitions for the temporal encoder, spatital encoder and final model encapsulating the both ####
import torch.nn as nn
import torch
import math 


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
    
### the architecture is inspired from https://www.sciencedirect.com/science/article/pii/S1746809423005633

# this is meant to extract the spatial information from the signal

class Spatial_Encoder(nn.Module): 
    def __init__(self,
                 num_heads = 8,
                 num_layers = 2,
                 input_features=32 # number of PSD features
                 ):
        
        
        super(Spatial_Encoder, self).__init__()
        
        self.enc_layer = nn.TransformerEncoderLayer(d_model=input_features, nhead=num_heads,batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.enc_layer,num_layers=num_layers)

    def forward(self, x):

        # x should be shape : batch, num_channels, num_psd_feature
        return self.transformer_encoder(x)
    

### the architecture is inspired from https://www.nature.com/articles/s41598-022-18502-3
class Temporal_Encoder(nn.Module): 
    def __init__(self,
                 num_heads = 8,
                 num_layers = 2,
                 input_features=64 # number of channels being used
                 ):
        
        
        super(Temporal_Encoder, self).__init__()
        
        self.enc_layer = nn.TransformerEncoderLayer(d_model=input_features, nhead=num_heads,batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.enc_layer,num_layers=num_layers)

    def forward(self, x):


        # x should be shape : batch, num_channels, signal_length
        return self.transformer_encoder(x)
###################################################################################

# used to merge the outputs of spatial and temporal encoders
class Classification_Head(nn.Module): 
    def __init__(self,dim_temp=64,dim_spatial=32,num_classes=109):
        
        
        super(Classification_Head, self).__init__()
        
        self.fc1 = nn.Linear(in_features=dim_temp+dim_spatial,out_features=num_classes)


        
    def forward(self, x):

        # x should be a tuple containg (tempral_outputs,spatial_outputs)

        x = torch.cat(tensors=x,dim=1)

        return self.fc1(x)    
    
##################################################################################

class EEG_Transformer(nn.Module):
    def __init__(self,num_classes = 109):

        super(EEG_Transformer,self).__init__()

        self.temporal_encoder = Temporal_Encoder()
        self.spatial_encoder = Spatial_Encoder()
        self.classification_head = Classification_Head(num_classes=num_classes)
    

    def forward(self,x):
        assert len(x) == 2 , f"input should be a tuple containing raw_eeg and psd_features"

        raw,psd = x
        # print("raw.shape , psd.shape ",raw.shape , psd.shape)






        info_token_psd = torch.rand(size=(psd.shape[0],1,psd.shape[-1]))
        psd = torch.cat(tensors=(info_token_psd,psd),dim=1)
        
        spatial_output = self.spatial_encoder(psd) 
        # output shape is batch , seq_len, features

        info_token_raw = torch.rand(size=(raw.shape[0],1,raw.shape[-1]))
        raw = torch.cat(tensors=(info_token_raw,raw),dim=1)
        temporal_output = self.temporal_encoder(raw)

        # print("raw.shape , psd.shape ",raw.shape , psd.shape)
        # print("temporal_output[:,0,:],spatial_output[:,0,:] ",temporal_output[:,0,:].shape,spatial_output[:,0,:].shape)


        return self.classification_head(x=(temporal_output[:,0,:],spatial_output[:,0,:]))
    

