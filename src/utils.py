"""
Utility functions for the project
"""

import yaml
from pathlib import Path

def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def test_config():
    """Test if config loads properly"""
    try:
        config = load_config()
        print("✅ Config loaded successfully!")
        print(f"\nDataset: {config['data']['dataset']}")
        print(f"Number of clients: {config['data']['num_clients']}")
        print(f"Model: {config['model']['architecture']}")
        return True
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False

if __name__ == "__main__":
    test_config()
