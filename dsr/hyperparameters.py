import copy
import os
import json

def create_configs_sensitivity(base_config):

    # set dataset to benchmark:
    base_config['task']['dataset']['name'] = 'benchmark'
    base_logdir = base_config['training']['logdir']

    # List of directories only, used to check if config is already completed
    dirlist = [x for x in os.listdir(base_logdir) if os.path.isdir(os.path.join(base_logdir, x))]

    # save base_config:
    config = copy.deepcopy(base_config)
    config['task']['name'] = 'base'
    config['training']['logdir'] = os.path.join(base_logdir, 'config_base')
    config_filename = os.path.join(base_logdir, "config_base.json")
    with open(config_filename, 'w') as f:
        json.dump(config, f, indent=4)

    # removed invalid weight!
    sensitivity = {
        'epsilon': {
            'locator': ['training', 'epsilon'],
            'range' : [0.5, 2]
        },
        'gtol_full': {
            'locator': ['training', 'optim_opt_full', 'gtol'],
            'range' : [0.1, 10]
        },
        'gtol_sub': {
            'locator': ['training', 'optim_opt_sub', 'gtol'],
            'range': [0.1, 10]
        },
        'maxiter_full': {
            'locator': ['training', 'optim_opt_full', 'maxiter'],
            'range': [0.5, 2]
        },
        'maxiter_sub': {
            'locator': ['training', 'optim_opt_sub', 'maxiter'],
            'range': [0.5, 2]},
        'num_layers': {
            'locator': ['controller', 'num_layers'],
            'range': [2]
        },
        'num_units': {
            'locator': ['controller', 'num_units'],
            'range': [0.5, 2]
        },
        'learning_rate': {
            'locator': ['controller', 'learning_rate'],
            'range': [0.5, 2]
        },
        'entropy_weight': {
            'locator': ['controller', 'entropy_weight'],
            'range': [0, 0.5, 2]
        },
        'batch_size': {
            'locator': ['training', 'batch_size'],
            'range': [0.5, 2]
        },
        'n_samples': {
            'locator': ['training', 'n_samples'],
            'range': [0.5, 2]
        },
        'max_length': {
            'locator': ['prior', 'length', 'max_'],
            'range': [0.5, 2]
        }
    }

    for var in sensitivity:
        locator = sensitivity[var]['locator']
        if len(locator) == 2:
            base_val = base_config[locator[0]][locator[1]]
        if len(locator) == 3:
            base_val = base_config[locator[0]][locator[1]][locator[2]]
        for factor in sensitivity[var]['range']:
            config = copy.deepcopy(base_config)

            new_val = base_val * factor
            if len(locator) == 2:
                config[locator[0]][locator[1]] = new_val
            if len(locator) == 3:
                config[locator[0]][locator[1]][locator[2]] = new_val

            config_name = f'{var}_{new_val}'
            config['task']['name'] = config_name
            # set correct logdir
            config['training']['logdir'] = os.path.join(base_logdir, config_name)


            # save_config
            if config_name not in dirlist:
                config_filename = os.path.join(base_logdir, config_name + '.json')
                with open(config_filename, 'w') as f:
                    json.dump(config, f, indent=4)