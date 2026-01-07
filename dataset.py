import torch
from torch.utils.data import Dataset,DataLoader
from tokenizer import Tokenizer
from datasets import load_from_disk
class IMDBDataset(Dataset):
    def __init__(self,path ,train_ratio = 0.2,mode='train'):
        super().__init__()
        dataset = load_from_disk(path)
        select_range = int(len(dataset['train'])*train_ratio)
       
        if(mode == "train"):
            
            selected_dataset = dataset['train'].select(range(select_range))
        elif (mode == "validation"):
            selected_dataset = dataset['train'].select(range(select_range,len(dataset['train'])))
        else:
            selected_dataset = dataset['test']
       
        self.text_datset = selected_dataset['text']
        self.label_dataset = selected_dataset['label']
      
    def __len__(self):
        return len(self.text_datset)
    def __getitem__(self,index)->tuple[str,int]:
        return(self.text_datset[index],self.label_dataset[index])