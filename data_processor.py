import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import yaml
import os
from typing import Tuple, Dict, List

class SpamDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: AutoTokenizer, max_length: int):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self) -> int:
        return len(self.labels)

class DataProcessor:
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Check CUDA availability and update device accordingly
        if self.config['training']['device'] == 'cuda' and not torch.cuda.is_available():
            print("CUDA is not available. Falling back to CPU.")
            self.config['training']['device'] = 'cpu'
            self.config['training']['mixed_precision'] = False
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['name'])
        self.device = torch.device(self.config['training']['device'])

    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the spam dataset."""
        try:
            df = pd.read_csv(self.config['paths']['data'])
            return df
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data loaders for training, validation, and testing."""
        df = self.load_data()
        
        # Split data
        train_df, temp_df = train_test_split(
            df, 
            test_size=self.config['data']['val_split'] + self.config['data']['test_split'],
            random_state=self.config['training']['seed']
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=self.config['data']['test_split'] / (self.config['data']['val_split'] + self.config['data']['test_split']),
            random_state=self.config['training']['seed']
        )

        # Create datasets
        train_dataset = SpamDataset(
            train_df['Message'].tolist(),
            train_df['Category'].map({'ham': 0, 'spam': 1}).tolist(),
            self.tokenizer,
            self.config['data']['max_length']
        )

        val_dataset = SpamDataset(
            val_df['Message'].tolist(),
            val_df['Category'].map({'ham': 0, 'spam': 1}).tolist(),
            self.tokenizer,
            self.config['data']['max_length']
        )

        test_dataset = SpamDataset(
            test_df['Message'].tolist(),
            test_df['Category'].map({'ham': 0, 'spam': 1}).tolist(),
            self.tokenizer,
            self.config['data']['max_length']
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers']
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers']
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers']
        )

        return train_loader, val_loader, test_loader

    def predict_single(self, text: str) -> torch.Tensor:
        """Prepare a single text for prediction."""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.config['data']['max_length'],
            return_tensors='pt'
        )
        return {k: v.to(self.device) for k, v in encoding.items()} 