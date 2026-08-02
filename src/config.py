import yaml
def load_config(path="config"):
    """
    读取yaml配置文件
    """
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config