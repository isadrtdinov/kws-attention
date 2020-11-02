def set_params():
    params = {
        # The most important parameter
        'random_seed': 112233,

        # System params
        'verbose': True,
        'num_workers': 8,

        # Wandb params
        'wandb_project': 'kws-attention',
        'num_examples': 5,

        # Data location
        'data_root': 'speech_commands/',

        # Checkpoints
        'checkpoint_dir': 'checkpoints/',
        'checkpoint_template': 'checkpoints/quartznet{}.pt',
        'model_checkpoint': 'checkpoints/quartznet30.pt',
        'load_model': True,

        # Data processing
        'valid_ratio': 0.2,
        'sample_rate': 16000,
        'num_mels': 64,
        'max_audio_length': 216000,
        'max_target_length': 200,

        # Augmentation params:
        'pitch_shift': 2.0, 'noise_scale': 0.005,
        'gain_db': (-10.0, 30.0), 'audio_scale': 0.15,

        # Optimizer params:
        'lr': 1e-3, 'weight_decay': 1e-3,
        'batch_size': 208, 'num_epochs': 30,
        'start_epoch': 31,
    }

    return params
