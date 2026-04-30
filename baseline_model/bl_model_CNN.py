import torch
import torch.nn as nn


class FrontCNN(nn.Module):
    def __init__(self, freq_bins, input_dim=128):
        super().__init__()

        reduced_f = freq_bins // 16

        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(
                128,
                input_dim,
                kernel_size=(reduced_f, 1),
                padding=0,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        x: (B, 2, F, T)
        """
            
        x = self.cnn(x)
        # x: (B, 128, 1, T)

        x = x.squeeze(2)
        # x: (B, 128, T)

        x = x.transpose(1, 2)
        # x: (B, T, 128)

        return x


