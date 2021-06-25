import logging

import yaml


logger = logging.getLogger("Model utilities")


def read_yaml(yaml_path):
    logger.debug(f"Loading YAML file at {yaml_path}")
    with open(yaml_path, 'r') as stream:
        try:
            json = yaml.safe_load(stream)
            logger.debug(f"YAML file at {yaml_path} loaded succesfully.")
            return json
        except yaml.YAMLError as exc:
            logger.error(exc)
            raise exc
