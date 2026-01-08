import torch
from torch.utils.data import Dataset,DataLoader
from tokenizer import Tokenizer
from datasets import load_from_disk
class IMDBDataset(Dataset):
    def __init__(self,path ,val_ratio = 0.2,mode='train'):
        super().__init__()
        dataset = load_from_disk(path)
        val_step = int(1/val_ratio)
        val_indices =list(range(0,len(dataset['train']),val_step))
        if(mode == "train"):
            all_indices = set (range(len(dataset['train'])))
            indices = sorted(list(all_indices - set(val_indices)))
            selected_dataset = dataset['train'].select(indices)
        elif (mode == "validation"):
            
            selected_dataset = dataset['train'].select(val_indices)
        else:
            selected_dataset = dataset['test']
       
        self.text_datset = selected_dataset['text']
        self.label_dataset = selected_dataset['label']
      
    def __len__(self):
        return len(self.text_datset)
    def __getitem__(self,index)->tuple[str,int]:
        return(self.text_datset[index],self.label_dataset[index])