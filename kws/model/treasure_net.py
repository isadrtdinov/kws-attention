from torch import nn
from .attention import MultiHeadAttention
from .crnn import CRNN


class TreasureNet(nn.Module):
    def __init__(self, num_keywords=1, time_steps=81, n_mels=40,
                 conv_channels=16, kernel_size=(20, 5), stride=(8, 2),
                 gru_hidden=256, gru_layers=2, num_heads=8, dropout=0.2):
        super(TreasureNet, self).__init__()
        self.encoder = CRNN(n_mels, conv_channels, kernel_size, stride,
                            gru_hidden, gru_layers, dropout)

        self.layer_norm = nn.LayerNorm(gru_hidden)
        self.attention = MultiHeadAttention(gru_hidden, num_heads, dropout)
        self.classifier = nn.Linear(self.encoder.time_frames * gru_hidden, num_keywords + 1)

    def forward(self, inputs):
        # inouts: (batch_size, time_steps, n_mels)

        outputs = self.encoder(inputs.unsqueeze(1))
        # outputs: (batch_size, time_frames, gru_hidden)

        outputs = self.layer_norm(outputs)
        outputs = self.attention(query=outputs, key=outputs, value=outputs)
        # outputs: (batch_size, time_frames, gru_hidden)

        outputs = outputs.view(outputs.shape[0], -1)
        # outputs: (batch_size, time_frames * gru_hidden)

        outputs = self.classifier(outputs)
        # outputs: (batch_size, num_keywords + 1)

        return outputs


def treasure_net(params):
    return TreasureNet(params['num_keywords'], params['time_steps'], params['n_mels'],
                       params['conv_channels'], params['kernel_size'], params['stride'],
                       params['gru_hidden'], params['gru_layers'], params['num_heads'],
                       params['dropout'])

