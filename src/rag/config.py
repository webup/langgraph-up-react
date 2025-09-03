import os
from types import SimpleNamespace
import threading
import yaml

# 用于确保配置只加载一次的锁和变量
_config_lock = threading.Lock()
_config_loaded = False
_config_data = None

def _load_config():
    """线程安全的配置加载函数"""
    global _config_loaded, _config_data
    
    if _config_loaded:
        return _config_data
    
    with _config_lock:
        if _config_loaded:
            return _config_data
            
        # Load config from YAML file
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yml')
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                _config_data = yaml.safe_load(file)
            _config_loaded = True
            return _config_data
        except FileNotFoundError:
            # 如果配置文件不存在，返回默认配置
            _config_data = {
                'RAGFLOW': {
                    'API_KEY': 'default-key',
                    'BASE_URL': 'http://localhost:8080',
                    'DATASET_ID': 'default-dataset'
                },
                'LLM': {
                    'API_KEY': 'default-key',
                    'BASE_URL': 'http://localhost:8080',
                    'MODEL': 'default-model',
                    'REWRITE_PROMPT': 'Rewrite: {user_input}',
                    'CHAT_PROMPT': 'Answer: {user_input}\nContext: {context}',
                    'MEMORY_PROMPT': 'Remember: {data}'
                },
                'RERANK_MODEL': {
                    'API_KEY': 'default-key',
                    'MODEL_NAME': 'default-rerank',
                    'BASE_URL': 'http://localhost:8080'
                },
                'AGENT': {}
            }
            _config_loaded = True
            return _config_data

# 延迟加载配置
def _get_config():
    return _load_config()

# 创建属性访问器
class _ConfigNamespace:
    def __init__(self, key):
        self._key = key
    
    def __getattr__(self, name):
        config = _get_config()
        return getattr(SimpleNamespace(**config[self._key]), name)

# 导出配置对象
RAGFLOW = _ConfigNamespace('RAGFLOW')
LLM = _ConfigNamespace('LLM')
RERANK_MODEL = _ConfigNamespace('RERANK_MODEL')
AGENT = _ConfigNamespace('AGENT')