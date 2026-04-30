import torch
import torch.nn as nn

class BEWOBackbone(nn.Module):
    """
    Both-Ears-Wide-Open (BEWO) Backbone
    Receives binaural log-mel spectrograms [Batch, 2, Freq, Time] and extracts Spatial Embeddings.
    """
    def __init__(self, input_freq=128, embed_dim=256):
        super().__init__()
        
        # Early Binaural Fusion CNN
        # Treats left and right ears as 2 input channels to explicitly learn binaural differences (ITD/ILD proxies) from the first layer
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None))  # Pool to 1 across the frequency dimension, keeping the time dimension
        )
        
        # Dimension reduction to the specified Embedding dimension
        self.proj = nn.Linear(256, embed_dim)

    def forward(self, x):
        """
        Input:
            x: [B, 2, F, T] (Binaural Spectrogram)
        Output:
            spatial_embedding: [B, embed_dim] (Global spatial representation for the entire audio clip)
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)  # [B, 256, 1, T']
        
        x = x.squeeze(2) # [B, 256, T']
        x = x.transpose(1, 2) # [B, T', 256]
        
        # Global Temporal Average Pooling
        # For linear probing, we need a single fixed-dimension embedding vector per clip
        spatial_embedding = x.mean(dim=1) # [B, 256]
        
        spatial_embedding = self.proj(spatial_embedding) # [B, embed_dim]
        
        return spatial_embedding

if __name__ == "__main__":
    # Test network forward pass
    dummy_input = torch.randn(8, 2, 128, 500) # [Batch, 2, 128 Mels, 500 TimeFrames]
    model = BEWOBackbone()
    output = model(dummy_input)
    print("BEWO Backbone Output Shape:", output.shape) # Should be [8, 256]
