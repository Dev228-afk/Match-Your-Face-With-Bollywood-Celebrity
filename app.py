# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 13:25:25 2022

@author: DEV
"""

from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from src.utils.all_utils import read_yaml, create_directory
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from keras.preprocessing.image import array_to_img
import streamlit as stm
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np

config= read_yaml('config/config.yaml') 
params= read_yaml('config/params.yaml') 

artifacts= config["artifacts"] # list of directories
artifacts_dir= artifacts["artifacts_dir"]

# Upload dir
upload_img_dir= artifacts['upload_img_dir']
# upload_path= os.path.join(artifacts, upload_img_dir)
upload_path= 'artifacts/upload/'

# Pickle format data dir
pickle_format_data_dir= artifacts["pickle_format_data_dir"]
img_pickle_file_name= os.path.join(artifacts_dir, pickle_format_data_dir)

raw_local_data_dir_path= os.path.join(artifacts_dir, pickle_format_data_dir) # raw_local_data_dir_path
pickle_file= os.path.join(raw_local_data_dir_path, img_pickle_file_name)

# Feature Path
feature_extraction_dir= artifacts['feature_extraction_dir']
extracted_feature_name= artifacts['extracted_features_name']

feature_extraction_path= os.path.join(artifacts_dir, feature_extraction_dir)
feature_name= os.path.join(feature_extraction_path, extracted_feature_name)

model_name= params['base']['BASE_MODEL']
include_top= params['base']['include_top']
pooling= params['base']['pooling']

detector= MTCNN()
model= VGGFace( model= model_name, include_top= include_top, 
                 input_shape=(224,224,3), pooling= pooling)
# print(pickle_file)
filenames= pickle.load(open('artifacts/pickle_format_data_dir/img_pkl_file.pkl', 'rb'))
feature_list= pickle.load(open(feature_name, 'rb'))

# Function For uploading image
def save_upload_img(upload_img):
    try:
        # create_directory(dirs= [upload_path])
        with open(os.path.join('artifacts/upload', upload_img.name), 'wb') as f:
            f.write(upload_img.getbuffer())
        
        return True 
    
    except:
        return False


# Feature Extractor for Uploaded Image
def extract_feature(img_path, models, detector):
    img= cv2.imread(img_path)
    result= detector.detect_faces(img)
    
    x, y, width, height= result[0]['box']
        
    face= img[y:y + height, x:x + width]
    
    # Extract Feature
    image= Image.fromarray(face)
    image= image.resize((224,224))
        
    face_array= np.asarray(image)
    face_array= face_array.astype('float32')
        
    img= np.expand_dims(face_array, axis=0)
    img= preprocess_input(img)
    result= models.predict(np.array(img)).flatten()
    return result
     

# Cosine Similarity Mesurement Function
def suggest(feature_list, feature):
    similarity= []
    
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(feature.reshape(1,-1), feature_list[i].reshape(1,-1))[0][0])
    
    index_pos= sorted( list(enumerate(similarity)), reverse=True,
                       key=lambda x: x[1])[0][0]    
    return index_pos
    
    
stm.title('To Whom You Look Like', anchor='https://github.com/Dev228-afk')
stm.markdown('**Let  Your Face Match with Bollywood Celebrities**.')

upload_img= stm.file_uploader('chose an image')

if upload_img is not None:
    # save img
        try:
            if save_upload_img(upload_img):
                # load img
                display_img= Image.open(upload_img)
            with stm.spinner('Wait for it...'):
                # extract features
                features= extract_feature(os.path.join(upload_path, upload_img.name),
                                          model ,detector)
               
                # suggestion
                index_pos= suggest(feature_list, features)
                predictor= " ".join(filenames[index_pos].split('\\')[1].split('_'))
            
                # display images
                col1, col2= stm.columns(2)
            stm.success('Done!')
            with col1:
                stm.header('Your Uploaded Image')
                stm.image(display_img,width=250)
                
            with col2:
                stm.header('Seems Like ' + predictor)
                stm.image(filenames[index_pos], width= 265)
        except:
            stm.warning('Please Upload Image with clear face')