import math
from torch import nn


class CRNN(nn.Module):
    def __init__(self, time_steps=81, num_mels=40,
                 conv_channels=16, kernel_size=(8, 4), stride=(2, 2),
                 gru_hidden=256, gru_layers=2, dropout=0.2):
        super(CRNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv_channels,
                               kernel_size=kernel_size, stride=stride)
        self.conv2 = nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels,
                               kernel_size=kernel_size, stride=stride)

        self.time_frames = time_steps
        self.num_features = num_mels

        for _ in range(2):
            self.time_frames = math.floor((self.time_frames - kernel_size[0]) / stride[0] + 1)
            self.num_features = math.floor((self.num_features - kernel_size[1]) / stride[1] + 1)

        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size=self.num_features * conv_channels, hidden_size=gru_hidden,
                          num_layers=gru_layers, dropout=dropout, batch_first=True)

    def forward(self, inputs, hidden=None):
        # inputs: (batch_size, 1, time_steps, num_mels)

        outputs = nn.functional.relu(self.conv1(inputs))
        outputs = self.dropout(outputs)
        outputs = nn.functional.relu(self.conv2(outputs))
        # outputs: (batch_size, conv_channels, time_frames, num_features)

        outputs = outputs.permute(0, 2, 3, 1)
        # outputs: (batch_size, time_frames, num_features, conv_channels)

        outputs = outputs.reshape(outputs.shape[0], outputs.shape[1], -1)
        outputs = self.dropout(outputs)
        # outputs: (batch_size, time_frames, num_features * conv_channels)

        outputs, hidden = self.gru(outputs)
        # outputs: (batch_size, time_frames, gru_hidden)
        # hidden: (batch_size, gru_layers, gru_hidden)

        return outputs, hidden

