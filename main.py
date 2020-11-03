import torch
import torchvision
from config import set_params
from asr.utils import set_random_seed, transforms
from asr.utils.data import SpeechCommandsDataset, load_data, split_data
from asr.model import treasure_net


def main():
    # set parameters and random seed
    params = set_params()
    set_random_seed(params['random_seed'])
    params['device'] = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    if params['verbose']:
        print('Using device', params['device'])

    # load and split data
    data = load_data(params['data_root'])
    train_data, valid_data = split_data(data, params['valid_ratio'])

    if params['verbose']:
        print('Data loaded and split')

    # create dataloaders
    train_transform = torchvision.transforms.Compose([
        transforms.RandomVolume(gain_db=params['gain_db']),
        transforms.RandomPitchShift(sample_rate=params['sample_rate'],
                                    pitch_shift=params['pitch_shift']),
        torchvision.transforms.RandomChoice([
            transforms.GaussianNoise(scale=params['noise_scale']),
            transforms.AudioNoise(scale=params['audio_scale'],
                                  sample_rate=params['sample_rate']),
        ]),
    ])

    train_dataset = SpeechCommandsDataset(root=params['data_root'], labels=train_data,
                                          keywords=params['keywords'], audio_seconds=params['audio_seconds'],
                                          sample_rate=params['sample_rate'], transform=train_transform)
    valid_dataset = SpeechCommandsDataset(root=params['data_root'], labels=valid_data,
                                          keywords=params['keywords'], audio_seconds=params['audio_seconds'],
                                          sample_rate=params['sample_rate'])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'],
                                               num_workers=params['num_workers'], shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=params['batch_size'],
                                               num_workers=params['num_workers'])

    if params['verbose']:
        print('Data loaders prepared')

    # initialize model and optimizer
    model = treasure_net(params).to(params['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    if params['load_model']:
        checkpoint = torch.load(params['model_checkpoint'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])

    if params['verbose']:
        print('Model and optimizer initialized')

    # create checkpoints folder
    if not os.path.isdir(params['checkpoint_dir']):
        os.mkdir(params['checkpoint_dir'])

    # initialize wandb
    if params['use_wandb']:
        wandb.init(project=params['wandb_project'])
        wandb.watch(model)

    # train
    train(model, optimizer, train_loader, valid_loader, params)


if __name__ == '__main__':
    main()

