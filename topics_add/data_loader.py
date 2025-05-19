import json
import numpy as np

class DataLoader:
    @staticmethod
    def load_config(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        if 'dtype' in config['lda_model']:
            config['lda_model']['dtype'] = getattr(np, config['lda_model']['dtype'])
        return config

    @staticmethod
    def load_meta(input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def save_json(data, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)