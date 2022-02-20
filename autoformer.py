import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange,repeat
from einops.layers.torch import Rearrange
import math

class LayerNorm(nn.Module):
    def __init__(self,nfeat):
        super().__init__()
        self.norm = nn.LayerNorm(nfeat)
    
    def forward(self,X):
        X_norm = self.norm(X)
        return X_norm - X_norm.mean(dim = 1).unsqueeze(dim = 1)

class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    
class AutoAttentionLayer(nn.Module):
    def __init__(self,nfeat,nhid,nhead, kernel_size = 25, dropout = 0.05):
        super().__init__()
        self.q = nn.Linear(nfeat,nhid*nhead)
        self.k = nn.Linear(nfeat,nhid*nhead)
        self.v = nn.Linear(nfeat,nhid*nhead)
        self.nhead = nhead
        self.out = nn.Linear(nhead*nhid,nfeat) 
        
        self.decomp1 = SeriesDecomp(kernel_size)
        self.decomp2 = SeriesDecomp(kernel_size)
        self.dropout = nn.Dropout(dropout) 
        
        self.ffn = nn.Sequential(
            Rearrange('b l f -> b f l'),
            nn.Conv1d(nfeat,2*nfeat,kernel_size = 3, padding=1, bias = False), 
            nn.Dropout(dropout),
            nn.ReLU(), 
            nn.Conv1d(2*nfeat,nfeat,kernel_size = 3, padding=1, bias = False),
            nn.Dropout(dropout),
            Rearrange('b f l -> b l f')
        )
        
    def forward(self,X,Y=None,Z=None,return_trend = False):
        self.topk = int(math.log(X.shape[1])+1) 
        
        if Y is None and Z is None:
            qkv = (self.q(X),self.k(X),self.k(X))
        else:
            qkv = (self.q(X),self.k(Y),self.v(Z))
        
        q,k,v = map(lambda x: rearrange(x,'b t (h f)-> b h f t',h = self.nhead),qkv)
        
        Tq,Tk = q.shape[-1],k.shape[-1]
        if Tq > Tk:
            v = F.pad(v,(0,Tq-Tk),"constant",0)
            k = F.pad(k,(0,Tq-Tk),"constant",0)
        else:
            v = v[:,:Tq]
            k = k[:,:Tq]

        #get time series autocorrelation
        q_fft = torch.fft.rfft(q,dim=-1)
        k_fft = torch.conj(torch.fft.rfft(k,dim=-1))
        lag = torch.fft.irfft(q_fft*k_fft)
        #use lag
        corr = lag.mean(dim=(1,2)) #use mean to accelerate as paper metions
        weights,delays = torch.topk(corr,self.topk,dim=-1)
        delays = torch.topk(weights.mean(dim=0),self.topk,dim=-1)[1]
        weights = torch.stack([weights[:,idx] for idx in delays],dim = -1)
        weights  = F.softmax(weights,dim=-1)
        #fusion
        values = v.clone()
        for idx in range(self.topk):
            values[:,:,:,idx] = torch.roll(v[:,:,:,idx],-int(delays[idx]),-1) * rearrange(weights[:,idx],"b ->b 1 1 ")

        values = self.out(rearrange(values,'b h f t -> b t (h f)'))

        X = X + self.dropout(values)

        #get residual (seasonality)
        y, trend1 = self.decomp1(X)
        y = self.ffn(y) #ffn 
        res,trend2 = self.decomp2(X+y)
        
        if not return_trend:
            return res
        else:
            return res,trend1+trend2
    
class Encoder(nn.Module):
    def __init__(self,nlayers = 2,nfeat = 2048,nhid = 256,nhead = 8, kernel_size = 25,dropout = 0.05):
        super().__init__()
        self.modulelist = nn.ModuleList([AutoAttentionLayer(nfeat,nhid,nhead,kernel_size,dropout) for i in range(nlayers)])
        self.norm = LayerNorm(nfeat)
    
    def forward(self,X):
        for m in self.modulelist:
            X = m(X)
        X = self.norm(X)
        return X

class DecoderLayer(nn.Module):
    def __init__(self,nfeat, nembed = 2048,nhid = 256,nhead = 8, kernel_size = 25,dropout = 0.05):
        super().__init__()
        self.self_attn = AutoAttentionLayer(nembed,nhid,nhead,kernel_size,dropout)
        self.cross_attn = AutoAttentionLayer(nembed,nhid,nhead,kernel_size,dropout)
        self.dropout = nn.Dropout(dropout)
        self.decomp = SeriesDecomp(kernel_size)
        self.projection = nn.Sequential(
                Rearrange('b t f-> b f t'),
                nn.Conv1d(in_channels=nembed, out_channels=nfeat, kernel_size=3, padding=1,
                                        padding_mode='circular', bias=False),
                Rearrange('b f t -> b t f'),
            )
        
    def forward(self,X,Y):
        X = X + self.dropout(self.self_attn(X))
        season1,trend1 = self.decomp(X)
        
        seanson2, trend2 = self.cross_attn(season1,Y,Y,return_trend = True)
        season = season1 + self.dropout(seanson2)
        
        trend = self.projection(trend1+trend2)
        # return X,trend
        return season,trend
    
class Decoder(nn.Module):
    def __init__(self,nlayers = 1,nfeat = 8 , nembed = 2048,nhid = 256,nhead = 8, kernel_size = 25,dropout = 0.05):
        super().__init__()
        self.modulelist = nn.ModuleList([DecoderLayer(nfeat,nembed,nhid,nhead,kernel_size,dropout) for i in range(nlayers)])
        self.norm = LayerNorm(nfeat)
        self.projection = nn.Sequential(
                Rearrange('b t f-> b f t'),
                nn.Conv1d(in_channels=nembed, out_channels=nfeat, kernel_size=3, padding=1,
                                        padding_mode='circular', bias=False),
                Rearrange('b f t -> b t f'),
            )
        
    def forward(self,X,Y,trend):
        for m in self.modulelist:
            X,residual_trend = m(X,Y)
            trend += residual_trend
        
        X = self.norm(self.projection(X))
        return X,trend

#There is actually no position embedding in origin paper
#But for a general usage, we use position embedding to avoid no time stamp input
class PosEmbedding(nn.Module):
    def __init__(self,nfeat,max_length = 20000):
        super().__init__()
        position = torch.arange(0, max_length).float().unsqueeze(1)
        div_term = (torch.arange(0, nfeat, 2).float() * -(math.log(10000.0) / nfeat)).exp()
        
        self.pe = torch.zeros(max_length, nfeat).float()
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)
        # self.register_buffer('pe', pe)
    def forward(self, x):
        return self.pe[:, :x.size(1),:].to(x.device)

#Temporal embedding can highly improve the performance
class TemporalEmbedding(nn.Module):
    def __init__(self, nfeat, max_size = 100, max_stamps = 20):
        super(TemporalEmbedding, self).__init__()
        self.max_stamps = max_stamps
        self.max_size = max_size
        
        self.embed_list = nn.ModuleList()
        for i in range(max_stamps):
            self.embed_list.append(nn.Embedding(max_size, nfeat))
        
    def forward(self, x):
        num_stamps = x.shape[-1]
        x = x.long()
        
        embeddings_list = []
        for i in range(num_stamps):
            embed = self.embed_list[i]
            embeddings_list.append(embed(x[:,:,i]))
        return sum(embeddings_list)

class AutoFormer(nn.Module):
    def __init__(self,pred_length = None ,encoder_layers=2,decoder_layers=1,nfeat = 8,nembed = 2048,nhid = 256,nhead = 8, kernel_size = 25,dropout = 0.05):
        '''
        pred_length `int`: 
            prediction time series length
       
        nfeat `int`:
           feature input and feature output
       
        nembed `int`:
           feature embedding length
       
        nhid `int`:
          hidden size in k,q,v
       
        nhead `int`:
          nhead in k,q,v
       
        kernel_size `int`:
          kernel_size in Moving Average
        '''
        super().__init__()
        self.enc_embedding = nn.Sequential(
                Rearrange('b t f-> b f t'),
                nn.Conv1d(in_channels=nfeat, out_channels=nembed, kernel_size=3, padding=1,
                                        padding_mode='circular', bias=False),
                Rearrange('b f t -> b t f'),
            )
        self.dec_embedding = nn.Sequential(
                Rearrange('b t f-> b f t'),
                nn.Conv1d(in_channels=nfeat, out_channels=nembed, kernel_size=3, padding=1,
                                        padding_mode='circular', bias=False),
                Rearrange('b f t -> b t f'),
            )
        self.trend_projection = nn.Linear(nembed,nfeat)
        self.seasonal_projection = nn.Linear(nembed,nfeat)
        
        self.encoder = Encoder(encoder_layers,nembed,nhid,nhead,kernel_size,dropout)
        self.decoder = Decoder(decoder_layers,nfeat,nembed,nhid,nhead,kernel_size,dropout)
        
        self.decomp = SeriesDecomp(kernel_size)
        
        self.pos_embedding = PosEmbedding(nembed)
        self.time_embedding = TemporalEmbedding(nembed)
        self.pred_length = pred_length
    
    def forward(self,X_enc,X_stamps = None,y_stamps = None):
        T = X_enc.shape[1]
        #decoder input preparation
        mean = torch.mean(X_enc, dim=1).unsqueeze(1).repeat(1, self.pred_length, 1)
        zeros = torch.zeros_like(mean,device = X_enc.device)
        
        #decomposition
        seasonal_init,trend_init = self.decomp(X_enc)
        trend_init = torch.cat([trend_init[:, -self.pred_length:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.pred_length:, :], zeros], dim=1)
        
        #encoder
        X_enc = self.enc_embedding(X_enc)
        if X_stamps is None:
            X_enc += self.pos_embedding(X_enc)
        else:
            X_enc += self.time_embedding(X_stamps)
            
        enc_out = self.encoder(X_enc)
        #decoder
        X_dec = self.dec_embedding(seasonal_init)
        if y_stamps is None:
            X_dec += self.pos_embedding(X_dec)
        else:
            X_dec[:,-self.pred_length:,:] = X_dec[:,-self.pred_length:,:] + self.time_embedding(y_stamps)
        
        seasonal_part,trend_part = self.decoder(X_dec,enc_out,trend_init)
        X = seasonal_part + trend_part
        X = X[:,-self.pred_length:,:]
        
        return X