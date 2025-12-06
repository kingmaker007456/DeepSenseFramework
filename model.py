import torch
import torch.nn as nn
import torch.nn.init as init
from config import Config

# --- General Utility for Initialization (New Enhancement) ---
def initialize_weights(model, init_method):
    """Initializes model weights using the specified method."""
    if init_method == 'none':
        return
        
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):
            if init_method == 'kaiming_uniform':
                init.kaiming_uniform_(module.weight, nonlinearity='relu')
            elif init_method == 'xavier_uniform':
                init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.constant_(module.bias, 0)
                
        elif isinstance(module, nn.Linear):
            if init_method == 'kaiming_uniform':
                init.kaiming_uniform_(module.weight, nonlinearity='relu')
            elif init_method == 'xavier_uniform':
                init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.constant_(module.bias, 0)
                
        elif isinstance(module, nn.BatchNorm1d):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)
            

# --- Feature Extraction Block ---
class TemporalCNN(nn.Module):
    """
    1D CNN block for multi-channel time-series feature extraction.
    Follows a two-layer CNN structure with Batch Normalization, ReLU, and Max Pooling.
    """
    def __init__(self, in_channels, out_channels, window_size):
        super(TemporalCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # Input: (Batch, Channels, Window_size)
            nn.Conv1d(in_channels, out_channels // 2, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(out_channels // 2),
            nn.ReLU(),
            nn.MaxPool1d(2), # Size / 2
            
            nn.Conv1d(out_channels // 2, out_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2) # Size / 4
        )
        self.output_feature_dim = (window_size // 4) * out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the input time-series data.
        Input shape: (Batch, Channels, Window_size)
        Output shape: (Batch, output_feature_dim)
        """
        x = self.conv_layers(x)
        # Flatten for fusion layer
        x = x.flatten(start_dim=1)
        return x # Output shape: (Batch, output_feature_dim)


# --- Multi-Head Attention Block (Standard Transformer Encoder Component) ---
class MultiHeadAttention(nn.Module):
    """
    Core Multi-Head Attention layer designed for a sequence length of 1 
    (as in feature fusion of a single window).
    """
    def __init__(self, input_dim, num_heads, dropout_rate=0.3):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert input_dim % num_heads == 0, "Input dimension must be divisible by num_heads"
        
        # Optimized QKV projection
        self.qkv_proj = nn.Linear(input_dim, input_dim * 3, bias=False)
        self.fc_out = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs Scaled Dot-Product Multi-Head Attention.
        Input x shape: (Batch, Sequence_length, Input_dim)
        Output shape: (Batch, Sequence_length, Input_dim)
        """
        batch_size = x.shape[0]
        
        # QKV Projection
        QKV = self.qkv_proj(x) 
        
        # Split into Q, K, V and reshape for multi-head: (B, H, S, D_head)
        Q, K, V = QKV.chunk(3, dim=-1) 
        
        Q = Q.reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Scaled Dot-Product Attention: (Q @ K_T) / sqrt(d_k)
        energy = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        attention = torch.softmax(energy, dim=-1)
        weighted_V = torch.matmul(attention, V) 

        # Concatenate heads and reshape back: (Batch, Sequence, Input_dim)
        weighted_V = weighted_V.permute(0, 2, 1, 3).contiguous()
        weighted_V = weighted_V.reshape(batch_size, -1, self.head_dim * self.num_heads)
        
        # Final linear layer
        output = self.fc_out(weighted_V)
        
        return output # Shape: (Batch, Sequence, Input_dim)

# --- Full Transformer Encoder Block for Fusion (New Enhancement) ---
class TransformerEncoderBlock(nn.Module):
    """
    Combines Multi-Head Attention (MHA), LayerNorm, Residual connections, 
    and a Feed-Forward Network (FFN). Acts as a feature fusion layer.
    """
    def __init__(self, input_dim, num_heads, ffn_hidden_dim, dropout_rate=0.3):
        super(TransformerEncoderBlock, self).__init__()
        
        # 1. Multi-Head Attention Sublayer
        self.attention = MultiHeadAttention(input_dim, num_heads, dropout_rate)
        self.norm1 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # 2. Position-wise Feed-Forward Network (FFN) Sublayer
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ffn_hidden_dim),
            nn.ReLU(), # Explicit ReLU activation for clarity
            nn.Dropout(dropout_rate),
            nn.Linear(ffn_hidden_dim, input_dim)
        )
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the input feature vector.
        Input x shape: (Batch, Input_dim)
        Output shape: (Batch, 1, Input_dim)
        """
        # Expand x to (Batch, Sequence=1, Input_dim) for MHA/Transformer operations
        x_mha = x.unsqueeze(1) 
        
        # 1. Attention Sublayer (Post-Norm style: Norm(x + Dropout(Sublayer(x))))
        _x_att = self.attention(x_mha) 
        x_mha = self.norm1(x_mha + self.dropout1(_x_att)) 
        
        # 2. FFN Sublayer
        _x_ffn = self.ffn(x_mha) 
        output = self.norm2(x_mha + self.dropout2(_x_ffn)) 

        return output # Shape: (Batch, 1, Input_dim)

# --- Complete DeepSense Model ---
class DeepSenseAttention(nn.Module):
    """
    The complete DeepSense model incorporating Multi-Branch CNN for feature 
    extraction, a Transformer Encoder Block for feature fusion (Attention), 
    a Bi-LSTM for temporal modeling, and a final classifier.
    """
    def __init__(self):
        super(DeepSenseAttention, self).__init__()
        
        # 1. Feature Extraction Branches
        self.accel_cnn = TemporalCNN(Config.ACCEL_CHANNELS, Config.CNN_OUT_CHANNELS, Config.WINDOW_SIZE)
        self.gyro_cnn = TemporalCNN(Config.GYRO_CHANNELS, Config.CNN_OUT_CHANNELS, Config.WINDOW_SIZE)
        
        # Calculate the total feature size after concatenation
        self.total_feature_size = self.accel_cnn.output_feature_dim + self.gyro_cnn.output_feature_dim

        # 2. Transformer-based Attention Fusion Layer (Enhanced)
        self.attention_fusion = TransformerEncoderBlock(
            self.total_feature_size, 
            Config.ATTENTION_HEADS,
            Config.FFN_HIDDEN_DIM,
            Config.DROPOUT_RATE
        )

        # 3. Temporal Modeling (Bi-LSTM) - Takes (B, Sequence=1, Features)
        self.lstm = nn.LSTM(
            input_size=self.total_feature_size,
            hidden_size=Config.LSTM_HIDDEN_SIZE,
            num_layers=Config.LSTM_NUM_LAYERS,
            bidirectional=True,
            batch_first=True
        )
        
        # 4. Final Classifier
        lstm_output_size = Config.LSTM_HIDDEN_SIZE * 2 # Bi-LSTM
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(64, Config.NUM_CLASSES)
        )
        
        # Initialize weights (New Enhancement)
        initialize_weights(self, Config.INIT_METHOD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the DeepSenseAttention model.
        Input x shape: (B, Window_size, Channels) 
        Output shape: (B, NUM_CLASSES)
        """
        # Separate and Permute for Conv1D: (B, Channels, Window_size)
        # Assume channels are ordered: [Accel_X, Accel_Y, Accel_Z, Gyro_X, Gyro_Y, Gyro_Z]
        accel_data = x[:, :, 0:Config.ACCEL_CHANNELS].permute(0, 2, 1)
        gyro_data = x[:, :, Config.ACCEL_CHANNELS:Config.TOTAL_CHANNELS].permute(0, 2, 1)
        
        # 1. Feature Extraction
        f_accel = self.accel_cnn(accel_data)
        f_gyro = self.gyro_cnn(gyro_data)
        
        # Fusion: Concatenate the features (B, total_feature_size)
        f_fused = torch.cat((f_accel, f_gyro), dim=1)
        
        # 2. Attention Fusion: (B, 1, total_feature_size)
        f_attention = self.attention_fusion(f_fused) 

        # 3. Temporal Modeling (LSTM expects 3D input (Batch, Sequence, Features))
        # Since we process windows independently, the sequence length is 1: (B, 1, Features)
        lstm_out, _ = self.lstm(f_attention) 

        # 4. Final Classification (Squeeze Sequence=1 dimension)
        output = self.classifier(lstm_out.squeeze(1))

        return output
