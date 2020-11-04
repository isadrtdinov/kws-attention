import math
from torch import nn


class CRNN(nn.Module):
    def __init__(self, time_steps=81, num_mels=40,
                 conv_channels=16, kernel_size=(20, 5), stride=(8, 2),
                 gru_hidden=256, gru_layers=2, dropout=0.2):
        super(CRNN, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=conv_channels,
                              kernel_size=kernel_size, stride=stride)

        self.time_frames = math.floor((time_steps - kernel_size[0]) / stride[0] + 1)
        self.num_features = math.floor((num_mels - kernel_size[1]) / stride[1] + 1)

        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size=self.num_features * conv_channels, hidden_size=gru_hidden,
                          num_layers=gru_layers, dropout=dropout, batch_first=True)

    def forward(self, inputs):
        # inputs: (batch_size, 1, time_steps, num_mels)

        outputs = nn.functional.relu(self.conv(inputs))
        time_frames = outputs.shape[2]
        # outputs: (batch_size, conv_channels, time_frames, num_features)

        outputs = outputs.permute(0, 2, 3, 1)
        # outputs: (batch_size, time_frames, num_features, conv_channels)

        outputs = outputs.reshape(outputs.shape[0], outputs.shape[1], -1)
        outputs = self.dropout(outputs)
        # outputs: (batch_size, time_frames, num_features * conv_channels)

        outputs, _ = self.gru(outputs)
        # outputs: (batch_size, time_frames, gru_hidden)

        return outputs

