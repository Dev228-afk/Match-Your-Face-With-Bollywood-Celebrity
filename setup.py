# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 16:49:14 2022

@author: DEV
"""

from setuptools import setup

setup(
      name= "src",
      version= "0.0.1.1",
      author= "Dev Patel",
      description= "Utility Package for Face Matching Project",
      author_email= "devansodariya55555@gmail.com",
      packages= ['src'],
      python_requirments= " >3.7 ",
      install_requirments= [
          'mtcnn==0.1.0',
          'tensorflow >=2.3.1',          
          'keras >2.4.3',
          'keras-vggface=0.6',          
          'keras_applicaions==1.0.8',
          'pyyaml',
          'tqdm',
          'scikit-learn',
          'streamlit',
          'bing-image-downloader'
          ],
      )

# Local Package