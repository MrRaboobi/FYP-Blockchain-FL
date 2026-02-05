"""
CNN-LSTM Model for ECG Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTM(nn.Module):
    """
    CNN-LSTM for ECG time-series classification
    """
    
    def __init__(
        self,
        input_size=360,
        num_classes=5,
        cnn_channels=[32, 64, 128],
        lstm_hidden_size=128,
        lstm_num_layers=2,
        dropout=0.5
    ):
        super(CNNLSTM, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # CNN Layers
        self.conv1 = nn.Conv1d(1, cnn_channels[0], kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(cnn_channels[0])
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(cnn_channels[1])
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(cnn_channels[1], cnn_channels[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(cnn_channels[2])
        self.pool3 = nn.MaxPool1d(2)
        
        # Calculate sequence length after pooling
        seq_len = input_size // 8  # 3 pooling layers (2^3 = 8)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels[2],
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0
        )
        
        # Fully connected
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_hidden_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: (batch_size, seq_len)
        Returns:
            (batch_size, num_classes)
        """
        # Add channel dimension: (batch, seq_len) -> (batch, 1, seq_len)
        x = x.unsqueeze(1)
        
        # CNN layers
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Reshape for LSTM: (batch, channels, seq_len) -> (batch, seq_len, channels)
        x = x.permute(0, 2, 1)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        x = hidden[-1]
        
        # Fully connected
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_num_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_model(config):
    """Create model from config"""
    model_config = config['model']
    
    model = CNNLSTM(
        input_size=config['data']['window_size'],
        num_classes=model_config['num_classes'],
        cnn_channels=model_config['cnn_channels'],
        lstm_hidden_size=model_config['lstm_hidden_size'],
        lstm_num_layers=model_config['lstm_num_layers'],
        dropout=model_config['dropout']
    )
    
    return model

def test_model():
    """Test model creation"""
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from utils import load_config
    
    config = load_config()
    model = create_model(config)
    
    print("=" * 60)
    print("Model Architecture")
    print("=" * 60)
    print(model)
    print("\n" + "=" * 60)
    print(f"Total parameters: {model.get_num_parameters():,}")
    print("=" * 60)
    
    # Test forward pass
    batch_size = 4
    seq_len = 360
    x = torch.randn(batch_size, seq_len)
    
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("\n✅ Model test passed!")

if __name__ == "__main__":
    test_model()
