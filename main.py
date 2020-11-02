import torch
from config import set_params


def main():
    # set parameters and random seed
    params = set_params()
    set_random_seed(params['random_seed'])
    params['device'] = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    if params['verbose']:
        print('Using device', params['device'])




if __name__ == '__main__':
    main()

