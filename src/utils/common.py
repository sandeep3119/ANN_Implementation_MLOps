import yaml

def readConfig(config_file):
    with open(config_file) as config:
        content=yaml.safe_load(config)
    return content