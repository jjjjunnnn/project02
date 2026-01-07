import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self,d_model=128,voc_size = 20000,hidden_size = 256,num_layer=2):
        super().__init__()
        self.embedding = nn.Embedding(voc_size,d_model) #change index to vector
        self.lstm = nn.LSTM(input_size = d_model,hidden_size = hidden_size,num_layers = num_layer,batch_first = True)
    
        self.linear = nn.Linear(hidden_size,1)
        self.sig = nn.Sigmoid()
    
    def forward(self,x,mask):
        x = self.embedding(x)
        output,(h_n,c_n) = self.lstm(x)
        last_hidden_state = h_n[-1]
        last_hidden_state = self.linear(last_hidden_state)
        b = self.sig(last_hidden_state)
        return b
        