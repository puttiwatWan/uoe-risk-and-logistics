import os
import yaml

from .config_scheme import Config

filename = "config.yaml"
if os.environ.get("CONFIG_FILENAME"):
    filename = os.environ.get("CONFIG_FILENAME")

with open(filename, "r") as f:
    config = Config(yaml.safe_load(f))
