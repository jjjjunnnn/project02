from datasets import load_from_disk
from collections import Counter
import re
import torch
class Tokenizer:
    def __init__(self,dataset = "~/workspace/project02/imdb_dataset",voc_size = 20000,max_length = 256):
        self.voc_size = voc_size
        self.dataset_add = dataset
        self.max_length = max_length
        self.pad_index = 0
        self.unk_index = 1
        self.w2i= {"<PAD>":0,"<UNK>":1} #word 2 index
     
        self.voc = []
    def preprocess(self,seq:list[str])->list[list[str]]: 
        return_list = []
        for s in seq:
            s = s.lower()
            s = re.sub(r"<br\s*/?>"," ",s)
            s = re.sub(r'([.,!?()":;])',r' \1 ',s)
            return_list.append(s.split())

        return return_list
    def make_voc(self):
        dataset = load_from_disk(self.dataset_add)
        train_text = dataset["train"]['text']
        train_text = self.preprocess(train_text)
        flaten = []
        for i in range(len(train_text)):
            flaten.extend(train_text[i])
        count = Counter(flaten)
        top_n_words = count.most_common(self.voc_size)
        i =2
        for w,f in top_n_words :
            self.w2i[w] = i
          
            i+=1

    def encode(self,word:str):
        return self.w2i.get(word,self.unk_index)
  
    
    def __call__(self,sequence:list[str])->tuple[torch.tensor,torch.tensor]:
        """
        Arg:
            input sequence(str)
        Return:
            (tokenized tensor, padding mask tensor)
            mask is 1 for <PAD>
            shape:(seq_num,256)
        """
        cliped = self.preprocess(sequence)
        
        for  i in range(len(cliped)):
            if(len(cliped[i]) >= self.max_length):
                cliped[i] = [self.encode(s) for s in cliped[i][:self.max_length]]
                
            else:
                cliped[i] = [self.encode(s) for s in cliped[i]]
                pad = [self.pad_index for i in range(self.max_length - len(cliped[i]))]
                cliped[i].extend(pad)
        cliped = torch.tensor(cliped)
        mask = (cliped == self.pad_index)
        return(cliped,mask.int())

        


    
    
    