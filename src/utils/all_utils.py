import yaml
import os
import logging

def read_yaml(path_to_file: str) -> dict:
    with open(path_to_file) as yaml_file:
        content= yaml.safe_load(yaml_file)
        
    return content

def create_directory(dairs: list) -> None:
    for dir_path in dairs:
        mkdir(dir_path, exist_ok=True)
        logging.info(f"Created directory: {dir_path}")