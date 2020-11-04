def set_params():
    params = {
        # The most important parameter
        'random_seed': 112233,

        # System params
        'verbose': True,
        'num_workers': 8,

        # Wandb params
        'use_wandb': True,
        'wandb_project': 'kws-attention',

        # Data location
        'data_root': 'speech_commands/',

        # Checkpoints
        'checkpoint_dir': 'checkpoints/',
        'checkpoint_template': 'checkpoints/treasure_net{}.pt',
        'model_checkpoint': 'checkpoints/treasure_net10.pt',
        'load_model': False,

        # Data processing
        'valid_ratio': 0.2,
        'fa_per_hour': 1.0,
        'audio_seconds': 1.0,
        'sample_rate': 16000,
        'time_steps': 81,
        'num_mels': 40,
        'keywords': ['marvin', 'sheila'],

        # Augmentation params:
        'pitch_shift': 2.0, 'noise_scale': 0.005,
        'gain_db': (-10.0, 30.0), 'audio_scale': 0.15,

        # Optimizer params:
        'lr': 1e-3, 'weight_decay': 1e-3,
        'batch_size': 512, 'num_epochs': 10,
        'start_epoch': 1,

        # TreasureNet params:
        'conv_channels': 16, 'kernel_size': (20, 5),
        'stride': (8, 2), 'gru_hidden': 256,
        'gru_layers': 2, 'num_heads': 8,
        'dropout': 0.2,
    }

    return params
