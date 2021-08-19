import json

import os

CONFIG_PATH = '/Users/mali/PycharmProjects/inductive_pykeen/best_configs'

paths = ('fully_inductive', 'semi_inductive')

KEYS_TO_DELETE = ['State', 'Notes', 'GPU Type', 'GPU Count', 'device']
KEYS_TO_REPLACE = {
    'loss_name': 'loss',
    'model_name': 'model',
}

VALUES_TO_REPLACE = {
    'MarginRankingLoss()': 'MarginRankingLoss',
    'margin': 'MarginRankingLoss',
    'BCEWithLogitsLoss()': 'BCEWithLogitsLoss',
}

for p in paths:
    path = os.path.join(CONFIG_PATH, p)
    for f in os.listdir(path):
        current_path = os.path.join(path, f)
        if not os.path.isfile(current_path):
            continue
        with open(current_path) as file:
            config = json.load(file)

        for key in KEYS_TO_DELETE:
            if key in config:
                del config[key]

        for key in KEYS_TO_REPLACE:
            if key in config:
                val = config[key]
                del config[key]
                config[KEYS_TO_REPLACE[key]] = val

        if config['loss'] in VALUES_TO_REPLACE:
            config['loss'] = VALUES_TO_REPLACE[config['loss']]

        if 'batch_size' not in config or 'embedding_dim' not in config:
            print(current_path)
        with open(current_path, 'w') as file:
            json.dump(config, file, indent=2, sort_keys=True)

