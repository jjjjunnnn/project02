import torch
import torch.nn as nn
import math
class FFN(nn.Module):
    def __init__(self,d_model1=512,d_model2 = 2048):
        super().__init__()
        self.activation = nn.GELU()
        self.linear1 = nn.Linear(d_model1,d_model2)
        self.linear2 = nn.Linear(d_model2,d_model1)
    def forward(self,x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,d_k,d_v,head_num,dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.head_num = head_num
        self.q_linear = nn.Linear(d_model,d_k *head_num,bias = False)
        self.k_linear = nn.Linear(d_model,d_k*head_num,bias = False)
        self.v_linear = nn.Linear(d_model,d_v*head_num,bias = False)
        self.last_linear = nn.Linear(head_num*d_v,d_model,bias = False)
        self.softmax = nn.Softmax(dim= -1)
        self.dropout = nn.Dropout(dropout)
    def split_head_dim(self,x):
        x = x.reshape(x.size(0),x.size(1),self.head_num,-1)
        x = x.transpose(1,2)
        return x
    def forward(self,x,mask):
        #x : (batch,seq_len,d_model)
        # mask shpae = (batch,seq_len = 256)
        mask = 1- mask # -> <PAD> is 0
        mask = mask.unsqueeze(1).unsqueeze(3) 
        seq_len = mask.size(2)
        mask = mask.repeat(1,self.head_num,1,seq_len) #(batch,head,seq_len,seq_len)
        mask = mask * (mask.transpose(-2,-1))
        q = self.q_linear(x) #(batch,seq_len,d_k*head)
        k = self.k_linear(x)
        v = self.v_linear(x)
        q = self.split_head_dim(q) #(batch,head,seq_len,d_k)
        k = self.split_head_dim(k)
        v = self.split_head_dim(v)
        qk =torch.matmul(q,k.transpose(-2,-1)) / math.sqrt(self.d_k) #(batch,head,seq_len,seq_len)
        if mask is not None:
            qk = qk.masked_fill(mask == 0,-1e9)
        heads = self.softmax(qk)
        heads = self.dropout(heads)
        heads = torch.matmul(heads,v)
        concated = heads.permute(0,2,1,3).contiguous()
        concated = concated.reshape(concated.size(0),concated.size(1),-1)
        concated = self.last_linear(concated)
        return concated

class Encoder(nn.Module):
    def __init__(self,d_model,multi_head_num,dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.ffn = FFN(d_model1=d_model,d_model2=2*d_model)
        self.multi_head_att = MultiHeadAttention(d_model,d_model//multi_head_num,d_model//multi_head_num,multi_head_num,dropout=dropout)
        self.dropout = nn.Dropout(dropout)
    def layer_norm(self,x):
        #x:(batch,seq_len,d_model)
        mean = x.mean(dim=-1,keepdim=True)
        std = torch.std(x,-1,keepdim = True)
        return (x-mean)/(std + 1e-6)
    def forward(self,x,mask):
        att = self.multi_head_att(x,mask)
        att = att + self.dropout(x)
        att = self.layer_norm(att)
        ffn = self.ffn(att)
        ffn = ffn + self.dropout(att)
        ffn = self.layer_norm(ffn)
        return ffn

class CustomTransformer(nn.Module):
    def __init__(self,d_model,dropout = 0.2,max_length = 256,voc_size = 20000,N=6,multi_num=8):
        super().__init__()
        print("model:VT")
        self.d_model = d_model
        self.max_length = max_length
        self.layer_num = N
        self.multi_head_num = multi_num
        self.embedding = nn.Embedding(voc_size ,d_model)
        self.layers = nn.ModuleList([Encoder(d_model,multi_num,dropout=dropout) for _ in range(N)])
        self.register_buffer('pe',self.pos_encoding())
        self.last_linear = nn.Linear(d_model,1)
        self.last_linear2 = nn.Linear(max_length,1)
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
    def pos_encoding(self):
        """
        Return: pe : (1,seq_len=256,d_model)
        """
        pe = torch.zeros(self.max_length,self.d_model)
        pos = torch.arange(self.max_length).unsqueeze(-1)
        i = torch.arange(start=0,end=self.d_model,step = 2).float() / self.d_model
        i = (10000**i)
        temp = pos / i
        pe[:,0::2] = torch.sin(temp)
        pe[:,1::2] = torch.cos(temp)
        return pe.unsqueeze(0)
    def forward(self,x,mask):
        # x: (batch,seq_len=256)
        x = self.embedding(x)
        # x: (batch , seq_len,d_model)
        x = x + self.pe
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x,mask)
        x = self.last_linear(x) #(batch,seq_len,1)
        x = x.squeeze(dim=2)
        x = self.last_linear2(x)
      
        return x