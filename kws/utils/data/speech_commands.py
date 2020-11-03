import torch
import torchaudio


class SpeechCommandsDataset(torch.utils.data.Dataset):
    def __init__(self, root, labels, keywords, audio_seconds=1.0,
                 sample_rate=16000, transform=None):
        super(SpeechCommandsDataset, self).__init__()
        self.root = root
        self.labels = labels
        self.max_audio_length = int(audio_seconds * sample_rate)
        self.sample_rate = sample_rate
        self.transform = transform
        self.category_id = {keyword: i for i, keyword in enumerate(keywords, 1)}

    def pad_sequence(self, sequence, max_length, fill=0.0, dtype=torch.float):
        padded_sequence = torch.full((max_length, ), fill_value=fill, dtype=dtype)
        sequence_length = min(sequence.shape[0], max_length)
        padded_sequence[:sequence_length] = sequence[:sequence_length]
        return padded_sequence

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        audio_info = self.labels.loc[index]
        waveform, sample_rate = torchaudio.load(audio_info.file)

        if sample_rate != self.sample_rate:
            raise ValueError('Wrong sample rate!')

        waveform = waveform.view(-1)
        waveform = waveform if self.transform is None else self.transform(waveform)
        waveform = self.pad_sequence(waveform, self.max_audio_length)

        target = self.category_id.get(audio_info.category, 0)
        return waveform, target

