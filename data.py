import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import requests
import os
from pathlib import Path
import pandas

class ChatbotDataset:
    def __init__(self, root, split, download=True, seed=42):
        self.root = Path(root)
        self.data_path = self.root / 'chatBotData.csv' 
        self.split = split
        self.seed = seed
        if download:
            ChatbotDataset.download_if_required(root)
        train_data, val_data = self.load_data()
        if split == 'train':
            self.data = train_data
        elif split == 'val':
            self.data = val_data
        else:
            raise ValueError('split should be either train or val')
    
    @staticmethod
    def download_if_required(root):
        root = Path(root)
        data_file_path = root / 'chatBotData.csv'
        if data_file_path.exists():
            return
        res = requests.get('https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv')
        root.mkdir(parents=True, exist_ok=True)
        with data_file_path.open('wb') as f:
            f.write(res.content)

    def load_data(self):
        df = pandas.read_csv(self.data_path)
        train_data = df.sample(frac=0.9, random_state=self.seed)
        val_data = df.drop(train_data.index)
        return train_data, val_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, key):
        row = self.data.iloc[key]
        return {
            'question': row['Q'],
            'answer': row['A']
        }

class ChatbotDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, tokenizer, max_length=128, data_dir='./chatbot_dataset', is_gpu=False):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.max_length = max_length
        self.pin_memory = is_gpu
        # 데이터를 모두 메모리로 미리 올리기 때문에 num_workers를 쓰면 오히려 overhead 때문에 더 느려짐
        # self.num_workers = os.cpu_count() if num_workers is None else num_workers

    def prepare_data(self):
        ChatbotDataset.download_if_required(self.data_dir)
    
    def setup(self, stage):
        self.train_dataset = self._preload_dataset('train')
        self.val_dataset = self._preload_dataset('val')

    def _preload_dataset(self, split):
        # tokenize & preload dataset into memory
        dataset = ChatbotDataset(root=self.data_dir, split=split, download=False)
        tokenized_samples = []
        for sample in dataset:
            enc_input = self.tokenizer(sample['question'], padding='max_length', max_length=self.max_length)
            dec_input = self.tokenizer(self.tokenizer.bos_token + sample['answer'], padding='max_length', max_length=self.max_length)
            dec_output = self.tokenizer.encode(sample['answer'] + self.tokenizer.eos_token, padding='max_length', max_length=self.max_length)
            tokenized_samples.append({
                'enc_x': torch.tensor(enc_input['input_ids']),
                'enc_x_padding_mask': torch.tensor(enc_input['attention_mask']),
                'dec_x': torch.tensor(dec_input['input_ids']),
                'dec_x_padding_mask': torch.tensor(dec_input['attention_mask']),
                'y': torch.tensor(dec_output)
            })
        return tokenized_samples

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, batch_size=self.batch_size, pin_memory=self.pin_memory)

    def teardown(self, stage):
        pass

    @staticmethod
    def preprocess_question(text, tokenizer):
        encoded_q = tokenizer.encode(text)
        return {
            'enc_x': torch.tensor([encoded_q]),
            'dec_x': torch.tensor([[tokenizer.bos_token_id]])
        }

