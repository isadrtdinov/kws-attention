import wandb
import torch
import torchaudio
import torchvision
from torch import nn
from ..utils.data import calculate_weights
from ..utils.transforms import SpectogramNormalize
from ..metrics.fnr_fpr import fnr_fpr_auc, fr_at_fa


def process_batch(model, optimizer, criterion, inputs, targets, params, train=True):
    optimizer.zero_grad()

    with torch.set_grad_enabled(train):
        logits = model(inputs)
        loss = criterion(loss, targets)

        if train:
            loss.backward()
            optimizer.step()

    probs = 1.0 - nn.functional.softmax(logits, dim=-1).numpy()
    targets = (targets != 0).long().numpy()

    auc = fnr_fpr_auc(probs, targets)
    fr = fr_at_fa(probs, targets, params['fa_per_hour'], params['audio_seconds'])

    return loss.item(), auc, fr


def process_epoch(model, optimizer, criterion, loader, spectrogramer, params, train=True):
    model.train() if train else model.eval()
    running_loss, running_auc, running_fr = 0.0, 0.0, 0.0

    for inputs, targets in loader:
        # convert waveforms to spectrograms
        with torch.no_grad():
            inputs = spectrogramer(inputs.to(params['device']))
            inputs = inputs.transpose(1, 2)

        # pass targets to device
        targets = targets.to(params['device'])

        loss, auc, fr = process_batch(model, optimizer, criterion, inputs, targets, params, train)
        running_loss += loss * inputs.shape[0]
        running_auc += auc * inputs.shape[0]
        running_fr += fr * inputs.shape[0]

    running_loss /= len(loader.dataset)
    running_auc /= len(loader.dataset)
    running_fr /= len(loader.dataset)

    return running_loss, running_auc, running_fr


def train(model, optimizer, train_loader, valid_loader, params):
    weights = calculate_weigths(train_loader.dataset.labels, params['keywords'])
    criterion = nn.CrossEntropyLoss(torch.tensor(weights))

    spectrogramer = torchvision.transforms.Compose([
        torchaudio.transforms.MelSpectrogram(
            sample_rate=params['sample_rate'],
            n_mels=params['num_mels'],
        ).to(params['device']),
        SpectogramNormalize(),
    ])

    for epoch in range(params['start_epoch'], params['num_epochs'] + params['start_epoch']):
        train_loss, train_auc, train_fr = process_epoch(model, optimizer, criterion, train_loader,
                                                        spectrogramer, params, train=True)

        valid_loss, valid_auc, valid_fr = process_epoch(model, optimizer, criterion, valid_loader,
                                                        spectrogramer, params, train=False)

        if params['use_wandb']:
            wand.log({'train loss': train_loss, 'train FNR/FPR-AUC': train_auc, 'train FR% @ FA/H = 1': train_fr,
                      'valid loss': valid_loss, 'valid FNR/FPR-AUC': valid_auc, 'valid FR% @ FA/H = 1': valid_fr})

        torch.save({
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
        }, params['checkpoint_template'].format(epoch))

