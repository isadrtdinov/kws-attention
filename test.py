import torch
import torchaudio
import torchvision
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from config import set_params
from kws.model import treasure_net
from kws.utils.transforms import SpectogramNormalize


def test():
    # set parameters
    params = set_params()
    params['device'] = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    if params['verbose']:
        print('Using device', params['device'])

    # initialize model
    model = treasure_net(params).to(params['device'])
    if params['load_model']:
        checkpoint = torch.load(params['model_checkpoint'])
        model.load_state_dict(checkpoint['model_state_dict'])

    # prepare test audio
    waveform, sample_rate = torchaudio.load(params['example_audio'])
    waveform = waveform[:1]

    if sample_rate != params['sample_rate']:
        waveform = waveform.squeeze(0).numpy()
        waveform = librosa.core.resample(waveform, sample_rate, params['sample_rate'])
        waveform = torch.from_numpy(waveform).unsqueeze(0)

    waveform = waveform.to(params['device'])
    spectrogramer = torchvision.transforms.Compose([
        torchaudio.transforms.MelSpectrogram(
            sample_rate=params['sample_rate'],
            n_mels=params['num_mels'],
        ).to(params['device']),
        SpectogramNormalize(),
    ])

    # calculate keyword probs
    spec = spectrogramer(waveform).transpose(1, 2)
    num_predicts = spec.shape[1] - params['time_steps']
    keyword_probs = np.zeros((num_predicts, len(params['keywords'])))
    hidden = None

    for i in range(num_predicts):
        with torch.no_grad():
            logits, hidden = model(spec[:, i:i + params['time_steps']], hidden)
            probs = torch.nn.functional.softmax(logits.detach(), dim=-1).cpu().numpy()

        keyword_probs[i] = probs[:, 1:]

    # plot results
    plt.figure(figsize=(12, 5))
    plt.rcParams.update({'font.size': 14})

    seconds_steps = np.linspace(0, waveform.shape[1] / params['sample_rate'], num_predicts)
    for i, keyword in enumerate(params['keywords']):
        plt.plot(seconds_steps, gaussian_filter1d(keyword_probs[:, i], sigma=3), label=keyword)

    plt.grid()
    plt.legend(title='keyword')
    plt.xlabel('time (s)')
    plt.ylabel('probability')
    plt.savefig(params['example_fig'])


if __name__ == '__main__':
    test()
