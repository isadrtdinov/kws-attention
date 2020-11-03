import random
import librosa
import torch
import torchaudio


class RandomPitchShift(object):
    def __init__(self, sample_rate=22050, pitch_shift=(-1.0, 1.0)):
        if isinstance(pitch_shift, (tuple, list)):
            self.min_pitch_shift = pitch_shift[0]
            self.max_pitch_shift = pitch_shift[1]
        else:
            self.min_pitch_shift = -pitch_shift
            self.max_pitch_shift = pitch_shift
        self.sample_rate=sample_rate

    def __call__(self, waveform):
        waveform = waveform.numpy()
        pitch_shift = random.uniform(self.min_pitch_shift, self.max_pitch_shift)
        waveform = librosa.effects.pitch_shift(waveform, sr=self.sample_rate,
                                               n_steps=pitch_shift)
        return torch.from_numpy(waveform)


class RandomVolume(object):
    def __init__(self, gain_db=(-50.0, 50.0)):
        self.gain = gain_db

    def __call__(self, waveform):
        rand_gain = random.uniform(self.gain[0], self.gain[1])
        return torch.clamp(torchaudio.functional.gain(waveform, rand_gain), -1.0, 1.0)


class AudioNoise(object):
    def __init__(self, scale=0.25, sample_rate=22050, examples=None):
        self.scale = scale
        self.sample_rate = sample_rate
        if examples is None:
            examples = ['brahms', 'choice', 'fishin', 'nutcracker', 'trumpet', 'vibeace']
            self.examples = []

            for example in examples:
                waveform, sample_rate = librosa.load(librosa.example(example))
                if sample_rate != self.sample_rate:
                    waveform = librosa.core.resample(waveform, sample_rate, self.sample_rate)
                self.examples.append(torch.from_numpy(waveform))
        else:
            self.examples = examples

    def __call__(self, waveform):
        noise = random.choice(self.examples)
        if noise.shape[0] < waveform.shape[0]:
            noise = noise.repeat(waveform.shape[0] // noise.shape[0] + 1)

        rand_pos = random.randrange(noise.shape[0] - waveform.shape[0] + 1)
        noise = noise[rand_pos:rand_pos + waveform.shape[0]]
        return waveform + self.scale * noise


class GaussianNoise(object):
    def __init__(self, scale=0.01):
        self.scale = scale

    def __call__(self, data):
        return data + self.scale * torch.randn(data.shape)


class SpectogramNormalize(object):
    def __init__(self, mean=-7.0, std=6.0, eps=1e-8):
        self.mean = mean
        self.std = std
        self.eps = 1e-8

    def __call__(self, spec):
        spec = torch.log(spec + self.eps)
        spec = (spec - self.mean) / self.std
        return spec

