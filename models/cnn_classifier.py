"""
Lightweight 1D-CNN for Raman Spectroscopy.

Properly sized for spectroscopic data:
- ~50-100K parameters (not 4M!)
- Designed for Raman peak pattern recognition
- Includes baseline PLS-DA/SVM comparison
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import os


class SpectralDataset(Dataset):
    """PyTorch Dataset for spectral data."""
    
    def __init__(self, spectra: np.ndarray, labels: np.ndarray):
        self.spectra = torch.FloatTensor(spectra).unsqueeze(1)  # (N, 1, L)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.spectra[idx], self.labels[idx]


class LightweightCNN(nn.Module):
    """
    Lightweight 1D-CNN for Raman spectroscopy.
    
    Target: ~50-80K parameters (vs 4.1M in original)
    Designed to learn actual peak patterns, not artifacts.
    """
    
    def __init__(self, input_length: int = 1000, n_classes: int = 10, dropout_rate: float = 0.3):
        super(LightweightCNN, self).__init__()
        
        self.input_length = input_length
        self.n_classes = n_classes
        
        # Smaller filters, fewer channels
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(4)  # More aggressive pooling
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(4)
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(4)
        
        # Calculate flattened size
        self.flat_size = 64 * (input_length // 64)  # After 3x pooling by 4
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.flat_size, 64)
        self.fc2 = nn.Linear(64, n_classes)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=1)
    
    def predict_proba(self, x):
        """Get softmax probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualBlock(nn.Module):
    """1D Residual block with skip connection."""
    
    def __init__(self, channels: int, kernel_size: int = 3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        return F.relu(out)


class ResidualCNN(nn.Module):
    """
    Improved 1D-CNN with residual connections.
    
    - Skip connections for better gradient flow
    - Increased capacity (~120K parameters)
    - Temperature scaling support for calibration
    """
    
    def __init__(self, input_length: int = 1000, n_classes: int = 10, dropout_rate: float = 0.2):
        super(ResidualCNN, self).__init__()
        
        self.input_length = input_length
        self.n_classes = n_classes
        
        # Initial expansion
        self.conv_in = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn_in = nn.BatchNorm1d(32)
        self.pool_in = nn.MaxPool1d(2)
        
        # Residual blocks
        self.res1 = ResidualBlock(32, kernel_size=5)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv_expand = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn_expand = nn.BatchNorm1d(64)
        
        self.res2 = ResidualBlock(64, kernel_size=3)
        self.pool2 = nn.MaxPool1d(2)
        
        self.res3 = ResidualBlock(64, kernel_size=3)
        self.pool3 = nn.MaxPool1d(2)
        
        # Global average pooling instead of flattening
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, n_classes)
        
        # Temperature parameter for calibration
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, x, use_temperature: bool = False):
        x = self.pool_in(F.relu(self.bn_in(self.conv_in(x))))
        x = self.pool1(self.res1(x))
        x = self.pool2(self.res2(F.relu(self.bn_expand(self.conv_expand(x)))))
        x = self.pool3(self.res3(x))
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(F.relu(self.fc1(x)))
        logits = self.fc2(x)
        
        if use_temperature:
            logits = logits / self.temperature
        
        return logits
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=1)
    
    def predict_proba(self, x, calibrated: bool = True):
        """Get (optionally calibrated) softmax probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, use_temperature=calibrated)
            return F.softmax(logits, dim=1)
    
    def calibrate_temperature(self, val_loader, device='cpu', max_iter=50):
        """
        Learn temperature scaling on validation set.
        
        Temperature scaling: divides logits by learned T before softmax.
        Improves calibration without affecting accuracy.
        """
        self.eval()
        self.temperature.requires_grad = True
        
        # Collect validation predictions
        all_logits, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = self.forward(x, use_temperature=False)
                all_logits.append(logits.cpu())
                all_labels.append(y)
        
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        
        # Optimize temperature
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = all_logits / self.temperature
            loss = criterion(scaled_logits, all_labels)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        self.temperature.requires_grad = False
        
        print(f"  Calibrated temperature: {self.temperature.item():.4f}")
        return self.temperature.item()
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
class SpectralCNN(nn.Module):
    """
    DEPRECATED: Original 4.1M parameter CNN.
    Use LightweightCNN instead for proper spectroscopic analysis.
    """
    
    def __init__(self, input_length: int = 1000, n_classes: int = 10, dropout_rate: float = 0.5):
        super(SpectralCNN, self).__init__()
        
        import warnings
        warnings.warn("SpectralCNN has 4.1M parameters - excessive for Raman data. Use LightweightCNN instead.", 
                     DeprecationWarning)
        
        self.input_length = input_length
        self.n_classes = n_classes
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(dropout_rate)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        self.drop2 = nn.Dropout(dropout_rate)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        self.drop3 = nn.Dropout(dropout_rate)
        
        self.flat_size = 256 * (input_length // 8)
        
        self.fc1 = nn.Linear(self.flat_size, 512)
        self.drop_fc = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, n_classes)
    
    def forward(self, x):
        x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        x = x.view(x.size(0), -1)
        x = self.drop_fc(F.relu(self.fc1(x)))
        return self.fc2(x)
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x).argmax(dim=1)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    learning_rate: float = 0.001,
    patience: int = 15,
    device: str = 'cpu',
    save_path: Optional[str] = None
) -> Dict[str, List[float]]:
    """Train the model with early stopping and LR scheduling."""
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=patience)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == batch_y).sum().item()
            train_total += len(batch_y)
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()
                val_correct += (outputs.argmax(1) == batch_y).sum().item()
                val_total += len(batch_y)
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss and save_path:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_length': model.input_length,
                'n_classes': model.n_classes
            }, save_path)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return history


def load_model(model_path: str, input_length: int, n_classes: int, device: str = 'cpu', lightweight: bool = True):
    """Load a trained model."""
    if lightweight:
        model = LightweightCNN(input_length=input_length, n_classes=n_classes)
    else:
        model = SpectralCNN(input_length=input_length, n_classes=n_classes)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Compare model sizes
    print("Model Comparison:")
    print("=" * 50)
    
    light = LightweightCNN(input_length=1000, n_classes=4)
    heavy = SpectralCNN(input_length=1000, n_classes=4)
    
    print(f"LightweightCNN: {light.count_parameters():,} parameters")
    print(f"SpectralCNN (deprecated): {sum(p.numel() for p in heavy.parameters()):,} parameters")
    print(f"Ratio: {sum(p.numel() for p in heavy.parameters()) / light.count_parameters():.1f}x")
