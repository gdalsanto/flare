import yaml
import argparse
from pathlib import Path
from flareverb.generate import fdn_dataset
from flareverb.config.config import BaseConfig, FDNConfig


def test_fdn_dataset(config):
    fdn_dataset(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FDN dataset generation.")
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to the configuration file.",
    )
    args = parser.parse_args()
    # resolve the relative file path
    file_path = Path(args.config_file).resolve()
    # read and parse the YAML file
    with open(file_path, "r") as file:
        config_dict = BaseConfig(**yaml.safe_load(file))
    test_fdn_dataset(config_dict)
