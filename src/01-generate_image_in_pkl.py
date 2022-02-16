import pickle
import numpy as np
import argparse
import os
from os import mkdir
import yaml
import logging

logging_str= "[%(asctime)s: %(levelname)s: %(module)s] %(message)s" 
log_dir= "logs"
os.mkdir(log_dir)
logging.basicConfig(filename= os.path.join(log_dir, "running_log.log"), level= logging.INFO, 
format= logging_str, filemode= "a") 

#-------------------------------------------------------------------
def read_yaml(path_to_file: str) -> dict:
    with open(path_to_file) as yaml_file:
        content= yaml.safe_load(yaml_file)
        
    return content

def create_directory(dirs: list) -> None:
    for dir_path in dirs:
        mkdir(dir_path)
        logging.info(f"Created directory: {dir_path}")

#-------------------------------------------------------------------

# Reading config file and generating data pkl file
def generate_data_pkl_file(config_file, params_path): 
    config= read_yaml(config_file) 
    params= read_yaml(params_path) 

    artifacts= config["artifacts"] # list of directories

    artifacts_dir= artifacts["artifacts_dir"]
    pickle_format_data_dir= artifacts["pickle_format_data_dir"]
    img_pkl_file_name= artifacts["img_pkl_file_name"]

    raw_local_data_dir_path= os.path.join(artifacts_dir, pickle_format_data_dir) # raw_local_data_dir_path
    create_directory(dirs=[raw_local_data_dir_path])

    pickle_file= os.path.join(raw_local_data_dir_path, img_pkl_file_name)

    data_path= params['base']['data_path']

    actors= os.listdir(data_path)
    file_name= []

    for actor in actors: # for listing actors name
        for _file in os.listdir(os.path.join(data_path, actor)): # for loading actor images
            file_name.append(os.path.join(data_path, actor, _file))  
    # print(len(file_name))

    logging.info(f"Total Actors are:{len(actors)}")
    logging.info(f"Total Actors images:{len(file_name)}")

    pickle.dump(file_name, open(pickle_file, 'wb'))

# Parsing arguments
if __name__ == "__main__": 
    parser= argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default='config/config.yaml', help= "Path to config file")
    parser.add_argument("-p", "--params", help= "Path to params file", default= "config/params.yaml")
    args= parser.parse_args()

    try:
        logging.info(f"----- Running Stage-01-----")
        generate_data_pkl_file(config_file= args.config, params_path= args.params)
        logging.info(f"----- Completed Stage-01-----")

    except Exception as e:
        logging.exception(e)
        raise e