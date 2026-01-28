import torch.nn as nn


class DeepLOB(nn.Module):
    def __init__(self, n_features: int, num_classes: int = 3):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 5), padding=(0, 2)), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(1, 5), padding=(0, 2)), nn.ReLU(),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0)), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(1, 0)), nn.ReLU(),
        )
        self.gru = nn.GRU(input_size=64 * n_features, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        B, T, F = x.shape
        x = x.unsqueeze(1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = x.permute(0, 2, 1, 3).reshape(B, T, -1)
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))
