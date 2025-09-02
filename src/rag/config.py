import os
from types import SimpleNamespace

import yaml

# Load config from YAML file
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yml')
with open(config_path, 'r', encoding='utf-8') as file:
    config_data = yaml.safe_load(file)

# Convert to object with dot notation access
RAGFLOW = SimpleNamespace(**config_data['RAGFLOW'])
LLM = SimpleNamespace(**config_data['LLM']) 
RERANK_MODEL = SimpleNamespace(**config_data['RERANK_MODEL'])
AGENT = SimpleNamespace(**config_data['AGENT'])