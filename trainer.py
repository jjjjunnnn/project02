from base_model import LSTM
from dataset import IMDBDataset
from tokenizer import Tokenizer
from transformer import Transformer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import os

class Trainer:
    def __init__(self, device, epochs, save_path, dataset_path="~/workspace/project02/imdb_dataset", batch=16, model="VT", load_path=None):
        self.device = device
        self.epochs = epochs
        self.save_path = save_path
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        self.early_stop_count = 0

        self.train_dataset = IMDBDataset(dataset_path, mode='train')
        self.validation_dataset = IMDBDataset(dataset_path, mode='validation')
        self.test_dataset = IMDBDataset(dataset_path, mode='test')
        
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch, collate_fn=self.collect_fn, shuffle=True)
        self.validation_dataloader = DataLoader(self.validation_dataset, batch_size=batch, collate_fn=self.collect_fn)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch, collate_fn=self.collect_fn)
        
        self.tokenizer = Tokenizer(dataset_path)
        self.model_name = model
        if self.model_name == "LSTM":
            self.model = LSTM(d_model=256, voc_size=20000, hidden_size=256, num_layer=2)
        elif self.model_name == "VT":
            self.model = Transformer(d_model=256, max_length=256, voc_size=20000, N=4, multi_num=8)
            
        self.model.to(device)
        self.loss = torch.nn.BCEWithLogitsLoss() 
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.01)

        if load_path:
            checkpoint = torch.load(load_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_loss = checkpoint['loss']
            self.history = checkpoint['history']
            print(f"Loaded checkpoint from {load_path} (Starting from epoch {self.start_epoch})")

    def collect_fn(self, batch):
        texts = [t[0] for t in batch]
        labels = [t[1] for t in batch]
        tokenized_vec, mask = self.tokenizer(texts)
      
        labels = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)
        return {"texts": tokenized_vec, "masks": mask, "labels": labels}

    def train(self, save_path_name="best_model.pth", early_stop=6):
            """
            Args:
                save_path_name (str): 저장할 체크포인트 파일의 이름 (확장자 포함)
                early_stop (int): 조기 종료를 위한 에폭 수
            """
            if not os.path.exists(self.save_path): 
                os.makedirs(self.save_path)
            
            for epoch in range(self.start_epoch, self.epochs):
                self.model.train()
                train_loss = 0
                pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
                for batch in pbar:
                    self.optimizer.zero_grad() 
                    input = batch['texts'].to(self.device)
                    mask = batch['masks'].to(self.device)
                    label = batch['labels'].to(self.device)
                    
                    output = self.model(input, mask)
                    loss = self.loss(output, label)
                    loss.backward()
                    self.optimizer.step()
                    
                    train_loss += loss.item()
                    pbar.set_postfix(loss=loss.item())
                
                v_loss = self.validate(epoch, train_loss)
                
                # 지정한 save_path_name으로 체크포인트 저장
                if v_loss < self.best_loss:
                    self.best_loss = v_loss
                    self.early_stop_count = 0
                    # "best_model.pth" 대신 인자로 받은 save_path_name 사용
                    self.save_checkpoint(epoch, v_loss, os.path.join(self.save_path, save_path_name))
                else:
                    self.early_stop_count += 1
                    if self.early_stop_count >= early_stop:
                        print(f"Early Stopping Triggered at Epoch {epoch+1}")
                        break

    def validate(self, epoch, train_loss):
        self.model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad(): 
            for batch in self.validation_dataloader:
                input, mask, label = batch['texts'].to(self.device), batch['masks'].to(self.device), batch['labels'].to(self.device)
                output = self.model(input, mask)
                val_loss += self.loss(output, label).item()
                
                # BCEWithLogitsLoss를 사용하므로 0.5가 아닌 0.0(로짓) 기준으로 판단
                correct += ((output > 0.0).float() == label).sum().item()
                total += label.size(0)
        
        avg_train = train_loss / len(self.train_dataloader)
        avg_val = val_loss / len(self.validation_dataloader)
        acc = 100 * correct / total
        self.history['train_loss'].append(avg_train)
        self.history['val_loss'].append(avg_val)
        self.history['val_acc'].append(acc)
        print(f"\nEpoch {epoch+1} Summary: Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val Acc: {acc:.2f}%")
        return avg_val

    def test(self):
        self.model.eval()
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc="Testing"):
                input, mask, label = batch['texts'].to(self.device), batch['masks'].to(self.device), batch['labels'].to(self.device)
                output = self.model(input, mask)
                test_loss += self.loss(output, label).item()
                correct += ((output > 0.5).float() == label).sum().item()
                total += label.size(0)
        print(f"\nTest Result: Loss: {test_loss/len(self.test_dataloader):.4f} | Acc: {100*correct/total:.2f}%")

    def save_checkpoint(self, epoch, loss, path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'history': self.history
        }, path)
