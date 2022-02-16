import pickle
import numpy as np
import argparse
import os
from os import mkdir
import yaml
import logging
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import tensorflow.python.keras.backend 
from keras.engine import  Model
from  tqdm import tqdm
from tensorflow.python.keras.layers import Layer, InputSpec

logging_str= "[%(asctime)s: %(levelname)s: %(module)s] %(message)s" 
log_dir= "logs"
# os.mkdir(log_dir, exist_ok=True)
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


def extractor(img_path,model):
    img= image.load_img(img_path, target_size=(224, 224))
    img_array= image.img_to_array(img)
    img= np.expand_dims(img_array, axis=0)
    img= preprocess_input(img)
    feature= model.predict(img)
    result= model.predict(img).flatten()

    return result

# Reading config file and generating data pkl file
def feature_extractor(config_file, params_path): 
    config= read_yaml(config_file) 
    params= read_yaml(params_path) 

    artifacts= config["artifacts"] # list of directories

    artifacts_dir= artifacts["artifacts_dir"]
    print('/n',artifacts_dir)
    pickle_format_data_dir= artifacts["pickle_format_data_dir"]
    img_pkl_file_name= artifacts["img_pkl_file_name"]
    
    img_pickle_file_name= os.path.join(artifacts_dir, pickle_format_data_dir, img_pkl_file_name)
    
    filenames= pickle.load(open('artifacts/pickle_format_data_dir/img_pkl_file.pkl', 'rb'))

    model_name= params['base']['BASE_MODEL']
    include_top= params['base']['include_top']
    pooling= params['base']['pooling']
    
    model= VGGFace( model= model_name, include_top= include_top, 
                    input_shape=(224,224,3), pooling= pooling )
    
    feature_extraction_dir= artifacts['feature_extraction_dir']
    extracted_feature_name= artifacts['extracted_features_name']
    
    feature_extraction_path= os.path.join(artifacts_dir, feature_extraction_dir)
    print(feature_extraction_path)
    feature_name= os.path.join(feature_extraction_path, extracted_feature_name)

    features = []
    for filename in tqdm(filenames):
        features.append(extractor(filename, model))
        
    pickle.dump(features, open('artifacts/extracted_features/feature_vectors.pkl', 'wb'))
    
    
    
if __name__ == "__main__": 
    parser= argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default='config/config.yaml', help= "Path to config file")
    parser.add_argument("-p", "--params", help= "Path to params file", default= "config/params.yaml")
    args= parser.parse_args()
    feature_extractor('config/config.yaml',
                      'config/params.yaml')
    try:
        logging.info(f"----- Running Stage-02-----")
        # generate_data_pkl_file(config_file= args.config, params_path= args.params)
        logging.info(f"----- Completed Stage-02-----")

    except Exception as e:
        logging.exception(e)
        raise e