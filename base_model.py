import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self,drop_out=0.2,d_model=256,voc_size = 20000,hidden_size = 256,num_layer=2):
        super().__init__()
        print("model:LSTM")
        self.embedding = nn.Embedding(voc_size,d_model) #change index to vector
        self.lstm = nn.LSTM(input_size = d_model,hidden_size = hidden_size,num_layers = num_layer,batch_first = True,dropout=drop_out,bidirectional=True)
    
        self.linear = nn.Linear(2*hidden_size,1)
    
    
    def forward(self,x,mask):
        length = mask.sum(dim=1).cpu()
        length = length -1
        x = self.embedding(x)
        output,(h_n,c_n) = self.lstm(x)
        batch_indices = torch.arange(output.size(0)).to(x.device)
        
        last_hidden_state = output[batch_indices,length]
        last_hidden_state = self.linear(last_hidden_state)
        
        return last_hidden_state
        