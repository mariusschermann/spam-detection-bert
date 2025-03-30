import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import yaml
import os
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)

class SpamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        # Convert numpy array to list for tokenizer
        texts = texts.tolist() if isinstance(texts, np.ndarray) else texts
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }
        return item

    def __len__(self):
        return len(self.labels)

class SpamClassifier(nn.Module):
    def __init__(self, config):
        super(SpamClassifier, self).__init__()
        
        # Store config
        self.config = config if isinstance(config, dict) else self._load_config(config)
        
        # Initialize the BERT model and tokenizer
        self.model_name = self.config['model']['name']
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.bert = AutoModel.from_pretrained(self.model_name)
        
        # Dropout layer
        self.dropout = nn.Dropout(self.config['model']['dropout'])
        
        # Classification head
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.config['model']['num_classes'])
        
        # Set device
        self.device = torch.device(self.config['training']['device'])
        self.to(self.device)
        
        # Log class distribution
        self.class_distribution = None
        
        # Initialize optimizer and scheduler
        self.optimizer = AdamW(self.parameters(), lr=self.config['training']['learning_rate'])
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=1000
        )
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.learning_rate = self.config['training']['learning_rate']
        self.min_learning_rate = 1e-6
        self.overfitting_threshold = 0.1  # 10% difference between train and val loss
    
    def _load_config(self, config_path):
        """Load config from file if path is provided."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass with additional spam features"""
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout for regularization
        pooled_output = self.dropout(pooled_output)
        
        # Get logits from classifier
        logits = self.classifier(pooled_output)
        
        return logits
    
    def adjust_learning_rate(self, train_loss, val_loss):
        """Dynamically adjust learning rate based on validation performance"""
        if val_loss > self.best_val_loss:
            self.patience_counter += 1
            if self.patience_counter >= 2:  # If validation loss increases for 2 consecutive checks
                self.learning_rate *= 0.5  # Reduce learning rate by half
                self.learning_rate = max(self.learning_rate, self.min_learning_rate)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate
                self.patience_counter = 0
        else:
            self.patience_counter = 0
            self.best_val_loss = val_loss

    def check_overfitting(self, train_loss, val_loss):
        """Check for overfitting and adjust dropout if necessary"""
        if val_loss > train_loss * (1 + self.overfitting_threshold):
            current_dropout = self.dropout.p
            if current_dropout < 0.4:  # Maximum dropout threshold
                new_dropout = min(current_dropout + 0.05, 0.4)
                self.dropout.p = new_dropout
                logging.info(f"Increasing dropout to {new_dropout} due to overfitting")

    def train_model(self):
        """Train the model with cross-validation"""
        # Load and preprocess data
        df = pd.read_csv(self.config['paths']['data'])
        df['label'] = df['Category'].map({'ham': 0, 'spam': 1})
        
        # Log initial class distribution
        total_samples = len(df)
        ham_samples = len(df[df['label'] == 0])
        spam_samples = len(df[df['label'] == 1])
        print(f"\nInitial class distribution:")
        print(f"Total samples: {total_samples}")
        print(f"Ham: {ham_samples} ({ham_samples/total_samples*100:.2f}%)")
        print(f"Spam: {spam_samples} ({spam_samples/total_samples*100:.2f}%)")
        
        # Show some example messages
        print("\nSample messages from each class:")
        print("\nHam examples:")
        for msg in df[df['label'] == 0]['Message'].head(3):
            print(f"- {msg[:100]}...")
        print("\nSpam examples:")
        for msg in df[df['label'] == 1]['Message'].head(3):
            print(f"- {msg[:100]}...")
        
        # Prepare data for stratified sampling
        texts = df['Message'].values
        labels = (df['Category'] == 'spam').astype(int).values
        
        # Calculate split sizes
        total_samples = len(texts)
        train_size = int(total_samples * self.config['data']['train_split'])
        val_size = int(total_samples * self.config['data']['val_split'])
        
        # Create stratified k-fold splitter for train/val/test
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Reduziert von 10 auf 5
        
        # Get indices for all splits
        splits = list(skf.split(texts, labels))
        
        # Get train indices (first 3 folds)
        train_idx = np.concatenate([splits[i][0] for i in range(3)])
        
        # Get validation indices (next fold)
        val_idx = splits[3][0]
        
        # Get test indices (last fold)
        test_idx = splits[4][0]
        
        # Create stratified datasets
        train_dataset = SpamDataset(texts[train_idx], labels[train_idx], 
                                  self.tokenizer, self.config['data']['max_length'])
        val_dataset = SpamDataset(texts[val_idx], labels[val_idx], 
                                self.tokenizer, self.config['data']['max_length'])
        test_dataset = SpamDataset(texts[test_idx], labels[test_idx], 
                                 self.tokenizer, self.config['data']['max_length'])
        
        # Log stratified split distribution
        logger.info(f"Training set distribution: {pd.Series(labels[train_idx]).value_counts(normalize=True)}")
        logger.info(f"Validation set distribution: {pd.Series(labels[val_idx]).value_counts(normalize=True)}")
        logger.info(f"Test set distribution: {pd.Series(labels[test_idx]).value_counts(normalize=True)}")
        
        # Log split sizes
        logger.info(f"Training set size: {len(train_idx)} ({len(train_idx)/total_samples*100:.1f}%)")
        logger.info(f"Validation set size: {len(val_idx)} ({len(val_idx)/total_samples*100:.1f}%)")
        logger.info(f"Test set size: {len(test_idx)} ({len(test_idx)/total_samples*100:.1f}%)")
        
        # Create data loaders with more workers
        train_loader = DataLoader(train_dataset, batch_size=self.config['training']['batch_size'], 
                                shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['training']['batch_size'], 
                              shuffle=False, num_workers=8, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config['training']['batch_size'], 
                               shuffle=False, num_workers=8, pin_memory=True)
        
        # Training loop with early stopping
        logger.info("Starting training...")
        for epoch in range(self.config['training']['epochs']):
            self.train()
            total_loss = 0
            correct = 0
            total = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config["training"]["epochs"]}')
            
            for batch_idx, batch in enumerate(progress_bar):
                self.optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct/total:.4f}'
                })
                
                # After each batch, check metrics
                if batch_idx % 100 == 0:  # Check every 100 batches
                    val_metrics = self.evaluate(val_loader)
                    val_loss = val_metrics[0]
                    val_acc = val_metrics[1]
                    self.adjust_learning_rate(total_loss / (batch_idx + 1), val_loss)
                    self.check_overfitting(total_loss / (batch_idx + 1), val_loss)
                    
                    # Early stopping check
                    if val_loss > self.best_val_loss:
                        self.patience_counter += 1
                        if self.patience_counter >= self.config['training']['early_stopping_patience']:
                            logging.info(f"Early stopping triggered after {epoch+1} epochs")
                            return test_acc, test_precision, test_recall, test_f1
                    else:
                        self.patience_counter = 0
                        self.best_val_loss = val_loss
                        self.save_model()
            
            avg_loss = total_loss / len(train_loader)
            accuracy = correct / total
            
            # Log metrics
            logger.info(f'Epoch {epoch+1}/{self.config["training"]["epochs"]}:')
            logger.info(f'Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.4f}')
            
            # Evaluate on test set
            test_metrics = self.evaluate(test_loader)
            test_loss, test_acc, test_precision, test_recall, test_f1 = test_metrics
            
            # Log test metrics
            logger.info(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
            logger.info(f'Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}')
            
            # Log final class distribution
            final_dist = pd.Series(predictions.cpu().numpy()).value_counts(normalize=True)
            logger.info(f"Final prediction distribution:\n{final_dist}")
            
            # Early stopping check
            if val_loss > self.best_val_loss:
                self.patience_counter += 1
                if self.patience_counter >= self.config['training']['early_stopping_patience']:
                    logging.info(f"Early stopping triggered after {epoch+1} epochs")
                    return test_acc, test_precision, test_recall, test_f1
            else:
                self.patience_counter = 0
                self.best_val_loss = val_loss
                self.save_model()
        
        return test_acc, test_precision, test_recall, test_f1
    
    def evaluate(self, data_loader):
        """Evaluate the model on the given data loader."""
        self.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total
        
        if len(all_preds) > 0:
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
            return avg_loss, accuracy, precision, recall, f1
        
        return avg_loss, accuracy

    def save_model(self):
        """Save the model to disk."""
        os.makedirs(self.config['paths']['models'], exist_ok=True)
        model_path = os.path.join(self.config['paths']['models'], 'best_model.pt')
        torch.save(self.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self):
        """Load the model from disk."""
        model_path = os.path.join(self.config['paths']['models'], 'best_model.pt')
        if os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path))
            logger.info(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"No model found at {model_path}")

    def predict(self, text):
        """Make a prediction for a single text with spam-specific features"""
        self.eval()
        with torch.no_grad():
            # Check for common spam indicators
            spam_indicators = [
                "urgent", "click here", "verify", "suspended", "account",
                "win", "prize", "offer", "limited time", "free",
                "money", "claim", "bank", "verify identity", "password"
            ]
            
            # Count spam indicators
            indicator_count = sum(1 for indicator in spam_indicators if indicator.lower() in text.lower())
            
            # Tokenize the text
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, 
                                  max_length=self.config['data']['max_length'])
            
            # Move inputs to device
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # Get model outputs
            logits = self(input_ids, attention_mask)
            
            # Adjust logits based on spam indicators
            if indicator_count >= 2:
                logits = logits + torch.tensor([[0.0, indicator_count * 0.5]]).to(self.device)
            
            # Get prediction
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            return predicted_class, confidence 